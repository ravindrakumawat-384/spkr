import logging
import os
import queue
import threading
import time
from typing import List

import numpy as np
import sounddevice as sd
import torch

from .config import (
    SAMPLE_RATE,
    CHANNELS,
    FRAMES_PER_BLOCK,
    INPUT_DEVICE,
    # new tuning/output options
    OVERLAP_MS,
    # normalization
    # flags
    # adaptive VAD and JSONL path will be added later if needed
    VAD_MAX_SPEECH_S,
    VAD_THRESHOLD,
    VAD_MIN_SPEECH_MS,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
    BEAM_SIZE,
)
from .audio import concat_and_maybe_empty, load_audio_file, rms_normalize
from .vad import load_vad_model, run_vad, cap_ts_bounds
from .model import load_model_safe, transcribe_chunk
from .diarize import get_or_load_pipeline, diarize_audio_data, assign_speakers, SpeakerIdMapper

logger = logging.getLogger("vad_transcriber")


class VADTranscriber:
    def __init__(self, model_path: str):
        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stop_event = threading.Event()
        self.model = load_model_safe(model_path)

        # VAD
        self.vad_model, self.get_speech_timestamps = load_vad_model()
        self.buffer: List[np.ndarray] = []
        self.min_speech_frames = int(SAMPLE_RATE * 0.5)  # at least 0.5s speech
        self._stream_thread = None
        self.speaker_mapper = SpeakerIdMapper()
        
        # Speaker diarization - load pipeline once at initialization
        self.diar_pipeline = None
        if os.getenv("HUGGINGFACE_TOKEN"):
            self.diar_pipeline = get_or_load_pipeline()
            if self.diar_pipeline:
                logger.info("Speaker diarization pipeline loaded successfully")

    def audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.debug(f"InputStream status: {status}")
        self.audio_queue.put(indata.copy())

    def start_recording(self) -> None:
        def _run():
            device = None
            if INPUT_DEVICE:
                try:
                    device = int(INPUT_DEVICE) if INPUT_DEVICE.isdigit() else INPUT_DEVICE
                except Exception:
                    device = INPUT_DEVICE
            try:
                logger.info(
                    f"Opening input stream (sr={SAMPLE_RATE}, ch={CHANNELS}, block={FRAMES_PER_BLOCK}, device={device if device is not None else 'default'})"
                )
                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    callback=self.audio_callback,
                    blocksize=FRAMES_PER_BLOCK,
                    device=device,
                ):
                    try:
                        dev_info = sd.query_devices(device or sd.default.device[0])
                        logger.info(f"Input device: {dev_info['name']} (index {dev_info['index']})")
                    except Exception:
                        pass
                    logger.info("🎙️ Listening with VAD... Ctrl+C to stop")
                    while not self._stop_event.is_set():
                        time.sleep(0.1)
            except Exception as e:
                logger.exception(f"Failed to open input stream: {e}")
                self._stop_event.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        self._stream_thread = t

    def run(self):
        self.start_recording()

        try:
            while not self._stop_event.is_set():
                try:
                    block = self.audio_queue.get(timeout=0.5)
                    self.buffer.append(block)
                except queue.Empty:
                    continue

                audio = concat_and_maybe_empty(self.buffer)
                logger.debug(f"Buffered audio frames: {audio.shape[0]}")

                speech_timestamps = run_vad(
                    audio,
                    self.vad_model,
                    self.get_speech_timestamps,
                    sample_rate=SAMPLE_RATE,
                    threshold=VAD_THRESHOLD,
                    min_speech_ms=VAD_MIN_SPEECH_MS,
                    min_silence_ms=VAD_MIN_SILENCE_MS,
                    pad_ms=VAD_SPEECH_PAD_MS,
                )

                if not speech_timestamps:
                    logger.debug("No speech detected in current buffer.")
                    continue

                max_frames = int(VAD_MAX_SPEECH_S * SAMPLE_RATE)

                logger.info(f"Detected {len(speech_timestamps)} speech region(s)")
                for idx, ts in enumerate(speech_timestamps, start=1):
                    start_i, end_i = cap_ts_bounds(ts, max_frames)
                    # apply small overlap padding
                    pad = int((OVERLAP_MS / 1000.0) * SAMPLE_RATE)
                    start_i = max(0, start_i - pad)
                    end_i = min(len(audio), end_i + pad)
                    if end_i - start_i < self.min_speech_frames:
                        logger.debug(f"Region {idx} too short: {end_i - start_i} frames")
                        continue

                    chunk = audio[start_i:end_i]
                    logger.debug(f"Region {idx} frames [{start_i}:{end_i}] -> {chunk.shape[0]} frames")

                    # Transcribe the chunk
                    segs = transcribe_chunk(
                        self.model,
                        chunk,
                        language="en",
                        beam_size=BEAM_SIZE,
                        condition_on_previous_text=True,
                    )
                    
                    # Apply speaker diarization if pipeline is available
                    if self.diar_pipeline is not None:
                        try:
                            diar_segments = diarize_audio_data(chunk, SAMPLE_RATE, pipeline=self.diar_pipeline)
                            assign_speakers(segs, diar_segments)
                        except Exception as e:
                            logger.debug(f"Real-time diarization failed for chunk {idx}: {e}")
                    
                    # Output the results
                    for seg in segs:
                        if getattr(seg, 'no_speech_prob', 0.0) > 0.6:
                            continue
                        text = seg.text.strip()
                        if text:
                            spk = getattr(seg, 'speaker', None)
                            if spk:
                                spk = self.speaker_mapper.map(spk)
                                logger.info(f"📝 [{spk}]: {text}")
                            else:
                                logger.info(f"📝 {text}")

                self.buffer = []

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt, stopping...")
            self._stop_event.set()
        except Exception as e:
            logger.exception(f"Error: {e}")
            self._stop_event.set()

    def stop(self):
        self._stop_event.set()
        try:
            if self._stream_thread and self._stream_thread.is_alive():
                self._stream_thread.join(timeout=2.0)
        except Exception:
            pass

    def run_file(self, file_path: str):
        """Transcribe a media file (wav/mp3/webm). Requires ffmpeg available in PATH."""
        try:
            logger.info(f"Transcribing file: {file_path}")
            segs = []
            with torch.no_grad():
                segments, _info = self.model.transcribe(
                    file_path,
                    language="en",
                    beam_size=BEAM_SIZE,
                    condition_on_previous_text=True,
                )
                for s in segments:
                    segs.append(s)

            # Diarization (optional, if token available & package installed)
            diar = []  # Default: no diarization
            if self.diar_pipeline is not None:
                try:
                    # Load the entire audio file into memory for diarization
                    full_audio = load_audio_file(file_path, SAMPLE_RATE)
                    # Run diarization on the full audio
                    diar = diarize_audio_data(full_audio, SAMPLE_RATE, pipeline=self.diar_pipeline)
                except Exception as e:
                    logger.debug(f"Full-file diarization failed: {e}")

            # Assign speaker labels if diarization was successful
            if diar:
                assign_speakers(segs, diar)

            # Output the results
            if not segs:
                logger.info("No speech detected in file.")
                return

            for s in segs:
                text = s.text.strip()
                if not text:
                    continue
                spk = getattr(s, 'speaker', None)
                if spk:
                    logger.info(f"[File][{spk}] {text}")
                else:
                    logger.info(f"[File] {text}")

        except Exception as e:
            logger.exception(f"File transcription error: {e}")
            
    def run_file_streaming(self, file_path: str):
        """Stream-like processing of a file: VAD-segment, transcribe immediately.
        Diarization runs on EACH CHUNK individually for immediate speaker labeling.
        Uses a globally cached pipeline for efficiency.
        """
        try:
            logger.info(f"Processing (streaming) file: {file_path}")
            audio = load_audio_file(file_path, SAMPLE_RATE)

            # VAD segmentation for the entire file
            timestamps = run_vad(
                audio,
                self.vad_model,
                self.get_speech_timestamps,
                sample_rate=SAMPLE_RATE,
                threshold=VAD_THRESHOLD,
                min_speech_ms=VAD_MIN_SPEECH_MS,
                min_silence_ms=VAD_MIN_SILENCE_MS,
                pad_ms=VAD_SPEECH_PAD_MS,
            )
            if not timestamps:
                logger.info("No speech detected.")
                return

            max_frames = int(VAD_MAX_SPEECH_S * SAMPLE_RATE)
            for idx, ts in enumerate(timestamps, start=1):
                start_i, end_i = cap_ts_bounds(ts, max_frames)
                if end_i - start_i < int(SAMPLE_RATE * 0.5):
                    continue
                chunk = audio[start_i:end_i]

                # Transcribe the chunk
                segs = transcribe_chunk(
                    self.model,
                    chunk,
                    language="en",
                    beam_size=BEAM_SIZE,
                    condition_on_previous_text=True,
                )

                # Run diarization ON THIS CHUNK ONLY, using the pre-loaded pipeline
                if self.diar_pipeline is not None:
                    try:
                        # Run diarization on this small chunk
                        diar_segments = diarize_audio_data(chunk, SAMPLE_RATE, pipeline=self.diar_pipeline)
                        # Assign speakers to the segments from this chunk
                        assign_speakers(segs, diar_segments)
                    except Exception as e:
                        logger.debug(f"Chunk-level diarization failed for chunk {idx}: {e}")

                # Output the results
                for s in segs:
                    text = s.text.strip()
                    if not text:
                        continue
                    spk = getattr(s, 'speaker', None)
                    if spk:
                        logger.info(f"[Stream][{idx}][{spk}] {text}")
                    else:
                        logger.info(f"[Stream][{idx}] {text}")

        except Exception as e:
            logger.exception(f"Streaming file processing error: {e}")
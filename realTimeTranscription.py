#!/usr/bin/env python3
"""
Realtime transcription with faster_whisper, *without* VAD.

Buffers audio, transcribes in fixed-duration chunks,
suppressing repeated transcripts to avoid spam.
"""

import os
import gc
import queue
import threading
import time
import logging
from typing import List, Optional

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("no_vad_transcriber")

# -----------------------
# Configuration
# -----------------------
MODEL_NAME = os.getenv("MODEL_NAME", "/home/dits403/models/whisper-small-ct2")
MODEL_COMPUTE = os.getenv("MODEL_COMPUTE", "float16")  # GPU friendly
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = int(os.getenv("CHANNELS", "1"))

BLOCK_SECONDS = float(os.getenv("BLOCK_SECONDS", "0.5"))
CHUNK_SECONDS = float(os.getenv("CHUNK_SECONDS", "2.0"))

BEAM_SIZE = int(os.getenv("BEAM_SIZE", "1"))

FRAMES_PER_BLOCK = int(SAMPLE_RATE * BLOCK_SECONDS)
FRAMES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_SECONDS)

# -----------------------
# Helper functions
# -----------------------

def free_cuda_cache():
    """Free memory and GPU cache to reduce risk of OOM."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def load_model_safe(path: str, try_gpu: bool = True) -> WhisperModel:
    """
    Load WhisperModel from `path`, try GPU if allowed; fallback to CPU.
    """
    free_cuda_cache()
    if try_gpu and torch.cuda.is_available():
        try:
            logger.info(f"Loading model '{path}' on CUDA (compute_type={MODEL_COMPUTE})")
            model = WhisperModel(path, device="cuda", compute_type=MODEL_COMPUTE)
            logger.info("Model loaded on CUDA")
            return model
        except Exception as e:
            logger.warning(f"CUDA load failed: {e}; falling back to CPU")
            free_cuda_cache()
    logger.info(f"Loading model '{path}' on CPU (compute_type=float32)")
    model = WhisperModel(path, device="cpu", compute_type="float32")
    return model

# -----------------------
# Main class
# -----------------------

class NoVADTranscriber:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        block_secs: float = BLOCK_SECONDS,
        chunk_secs: float = CHUNK_SECONDS,
        model_path: str = MODEL_NAME,
        beam_size: int = BEAM_SIZE,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_secs = block_secs
        self.chunk_secs = chunk_secs
        self.model_path = model_path
        self.beam_size = beam_size

        self.frames_per_block = int(self.sample_rate * self.block_secs)
        self.frames_per_chunk = int(self.sample_rate * self.chunk_secs)

        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._audio_buffer: List[np.ndarray] = []

        self._stop_event = threading.Event()
        self._prev_transcript: Optional[str] = None

        self.model = load_model_safe(self.model_path, try_gpu=True)

    def audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.debug(f"InputStream status: {status}")
        # Make a copy so sounddevice doesn't reuse buffer memory
        self.audio_queue.put(indata.copy())

    def start_recording(self) -> None:
        def _run():
            try:
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    callback=self.audio_callback,
                    blocksize=self.frames_per_block
                ):
                    logger.info(f"Audio input started (sr={self.sample_rate}, channels={self.channels})")
                    while not self._stop_event.is_set():
                        time.sleep(0.1)
            except Exception as e:
                logger.exception(f"Input stream error: {e}")
                self._stop_event.set()

        threading.Thread(target=_run, daemon=True).start()

    def _consume_buffer(self) -> Optional[np.ndarray]:
        """
        If buffer has enough frames for one chunk, return that chunk as
        a float32 1D numpy array; reset buffer to leftover.
        """
        total_frames = sum(b.shape[0] for b in self._audio_buffer)
        if total_frames < self.frames_per_chunk:
            return None

        concatenated = np.concatenate(self._audio_buffer, axis=0)
        needed = concatenated[: self.frames_per_chunk]
        remainder = concatenated[self.frames_per_chunk :]

        self._audio_buffer = [remainder] if remainder.size else []

        # convert to mono if needed
        if needed.ndim > 1 and needed.shape[1] > 1:
            needed = needed.mean(axis=1)
        needed = needed.flatten().astype(np.float32)

        return needed

    def run(self) -> None:
        self.start_recording()
        logger.info(f"Starting transcription loop: chunk size {self.chunk_secs}s, block {self.block_secs}s")

        try:
            while not self._stop_event.is_set():
                # get block(s)
                try:
                    block = self.audio_queue.get(timeout=0.5)
                    self._audio_buffer.append(block)
                except queue.Empty:
                    continue

                chunk = self._consume_buffer()
                if chunk is None:
                    continue

                # transcribe
                with torch.no_grad():
                    segments, info = self.model.transcribe(
                        chunk, beam_size=self.beam_size, language="en"
                    )

                # join segment texts
                transcript = " ".join(s.text.strip() for s in segments).strip()
                if not transcript:
                    continue

                # suppress repeats
                if transcript == self._prev_transcript:
                    logger.debug("Skipping duplicate transcript.")
                else:
                    logger.info(f"TRANSCRIPT: {transcript}")
                    self._prev_transcript = transcript

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt; stopping.")
            self._stop_event.set()
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            self._stop_event.set()

    def stop(self) -> None:
        self._stop_event.set()

# -----------------------
# Entrypoint
# -----------------------

if __name__ == "__main__":
    trans = NoVADTranscriber()
    trans.run()

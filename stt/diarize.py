import logging
import os
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from io import BytesIO
import soundfile as sf

logger = logging.getLogger("vad_transcriber")


def _import_pyannote():
    try:
        from pyannote.audio import Pipeline  # type: ignore
        return Pipeline
    except Exception as e:
        logger.warning(f"pyannote.audio not available: {e}")
        return None


# --- NEW: Global variable to hold the loaded pipeline ---
_LOADED_PIPELINE = None

def get_or_load_pipeline() -> Optional[Any]:
    """
    Returns a loaded pyannote pipeline, loading it once if not already loaded.
    Returns None if loading fails or token is not set.
    """
    global _LOADED_PIPELINE

    if _LOADED_PIPELINE is not None:
        return _LOADED_PIPELINE

    Pipeline = _import_pyannote()
    if Pipeline is None:
        return None

    token = os.getenv("HUGGINGFACE_TOKEN", "")
    if not token:
        logger.warning("HUGGINGFACE_TOKEN not set; cannot load diarization pipeline")
        return None

    try:
        logger.info("Loading pyannote/speaker-diarization pipeline...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=token)
        _LOADED_PIPELINE = pipeline
        logger.info("Diarization pipeline loaded successfully.")
        return pipeline
    except Exception as e:
        logger.exception(f"Failed to load diarization pipeline: {e}")
        return None


def diarize_audio_data(
    audio_data: np.ndarray,
    sample_rate: int,
    pipeline: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Run speaker diarization on a numpy audio array.
    `audio_data` should be a 1D numpy array of float32 samples.
    If `pipeline` is None, it will try to load/get the global one.
    """
    if pipeline is None:
        pipeline = get_or_load_pipeline()
        if pipeline is None:
            return []

    try:
        # Create an in-memory WAV file from the numpy array
        buffer = BytesIO()
        # pyannote expects int16, so we scale the float32 [-1, 1] data
        sf.write(buffer, (audio_data * 32767).astype(np.int16), sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)

        # Run diarization
        diarization = pipeline(buffer)
        results: List[Dict[str, Any]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            results.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            })
        return results
    except Exception as e:
        logger.debug(f"Diarization on audio data failed: {e}")
        return []


def assign_speakers(whisper_segments, diar_segments: List[Dict[str, Any]]):
    """
    Assign speaker labels to whisper segments by maximum time overlap with diarization segments.
    Modifies segments in-place to add .speaker attribute if possible.
    """
    if not diar_segments:
        return whisper_segments

    def overlap(a_start, a_end, b_start, b_end):
        return max(0.0, min(a_end, b_end) - max(a_start, b_start))

    for seg in whisper_segments:
        try:
            s_start = float(getattr(seg, 'start', 0.0))
            s_end = float(getattr(seg, 'end', s_start))
        except Exception:
            continue

        best_speaker = None
        best_ov = 0.0
        for d in diar_segments:
            ov = overlap(s_start, s_end, d["start"], d["end"])
            if ov > best_ov:
                best_ov = ov
                best_speaker = d["speaker"]
        if best_speaker is not None and best_ov > 0:
            try:
                setattr(seg, 'speaker', best_speaker)
            except Exception:
                pass
    return whisper_segments


class SpeakerIdMapper:
    """Stable mapping of diarization labels to SPEAKER_01.. per session."""
    def __init__(self):
        self._map: Dict[str, str] = {}
        self._next = 1

    def map(self, label: Optional[str]) -> Optional[str]:
        if not label:
            return None
        if label in self._map:
            return self._map[label]
        pretty = f"SPEAKER_{self._next:02d}"
        self._map[label] = pretty
        self._next += 1
        return pretty
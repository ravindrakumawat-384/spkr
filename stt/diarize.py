import logging
import os
from typing import List, Dict, Any

logger = logging.getLogger("vad_transcriber")


def _import_pyannote():
    try:
        from pyannote.audio import Pipeline  # type: ignore
        return Pipeline
    except Exception as e:
        logger.warning(f"pyannote.audio not available: {e}")
        return None


def diarize_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Run speaker diarization on a media file using pyannote.audio, if available.
    Requires env HUGGINGFACE_TOKEN for the pretrained pipeline.
    Returns a list of dicts: {start: float, end: float, speaker: str}
    """
    Pipeline = _import_pyannote()
    if Pipeline is None:
        return []

    token = os.getenv("HUGGINGFACE_TOKEN", "")
    if not token:
        logger.warning("HUGGINGFACE_TOKEN not set; skipping diarization")
        return []

    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=token)
        diarization = pipeline(file_path)
        results: List[Dict[str, Any]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            results.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            })
        return results
    except Exception as e:
        logger.exception(f"Diarization failed: {e}")
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



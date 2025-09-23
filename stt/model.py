import gc
import logging
import torch
from faster_whisper import WhisperModel
from .config import MODEL_COMPUTE

logger = logging.getLogger("vad_transcriber")


def free_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def load_model_safe(path: str) -> WhisperModel:
    free_cuda_cache()
    if torch.cuda.is_available():
        try:
            logger.info(f"Loading Whisper model on CUDA (compute={MODEL_COMPUTE})")
            return WhisperModel(path, device="cuda", compute_type=MODEL_COMPUTE)
        except Exception as e:
            logger.warning(f"CUDA load failed: {e}, falling back to CPU")
    logger.info("Loading Whisper model on CPU (float32)")
    return WhisperModel(path, device="cpu", compute_type="float32")


def transcribe_chunk(
    model: WhisperModel,
    chunk,
    language: str,
    beam_size: int,
    condition_on_previous_text: bool = True,
    suppress_tokens = list,
):
    if chunk.size == 0:
        return []
    with torch.no_grad():
        segments, _info = model.transcribe(
            chunk,
            language=language,
            beam_size=beam_size,
            condition_on_previous_text=condition_on_previous_text,
            suppress_tokens = None
        )
    return segments



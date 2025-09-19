import logging
import torch

logger = logging.getLogger("vad_transcriber")


def load_vad_model():
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    # Keep on CPU to avoid device mismatch with numpy inputs via utils
    model = model.cpu().eval()
    (get_speech_timestamps, _, _read_audio, _, _collect_chunks) = utils
    return model, get_speech_timestamps


def run_vad(
    audio,
    vad_model,
    get_speech_timestamps,
    *,
    sample_rate: int,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    pad_ms: int,
):
    if audio.size == 0:
        return []
    return get_speech_timestamps(
        audio,
        vad_model,
        sampling_rate=sample_rate,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=pad_ms,
        return_seconds=False,
    )


def cap_ts_bounds(ts, max_frames: int):
    start_i = ts['start']
    end_i = ts['end']
    if end_i - start_i > max_frames:
        end_i = start_i + max_frames
    return start_i, end_i



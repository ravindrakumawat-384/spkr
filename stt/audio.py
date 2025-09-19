from typing import List
import numpy as np
import torchaudio
import torch


def ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio is None:
        return np.array([], dtype=np.float32)
    if audio.ndim > 1 and audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    audio = audio.flatten().astype(np.float32, copy=False)
    return audio


def concat_and_maybe_empty(buffers: List[np.ndarray]) -> np.ndarray:
    if not buffers:
        return np.array([], dtype=np.float32)
    cat = np.concatenate(buffers, axis=0)
    return ensure_mono_float32(cat)


def load_audio_file(file_path: str, target_sr: int) -> np.ndarray:
    """Load audio file and return mono float32 numpy at target_sr."""
    wav, sr = torchaudio.load(file_path)
    # to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    # to numpy float32
    return wav.detach().cpu().to(torch.float32).numpy()


def rms_normalize(audio: np.ndarray, target_rms: float) -> np.ndarray:
    if audio.size == 0:
        return audio
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms <= 1e-8:
        return audio
    gain = target_rms / rms
    return np.clip(audio * gain, -1.0, 1.0)



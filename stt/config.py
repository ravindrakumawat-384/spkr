import os

# Model
MODEL_PATH = os.getenv("MODEL_PATH", "/home/dits403/models/whisper-small-ct2")
MODEL_COMPUTE = os.getenv("MODEL_COMPUTE", "float16")

# Audio
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = int(os.getenv("CHANNELS", "1"))
BLOCK_SECONDS = float(os.getenv("BLOCK_SECONDS", os.getenv("BLOCK_SECS", "1.5")))
FRAMES_PER_BLOCK = int(SAMPLE_RATE * BLOCK_SECONDS)
INPUT_DEVICE = os.getenv("INPUT_DEVICE", "")
OVERLAP_MS = int(os.getenv("OVERLAP_MS", "150"))  # overlap around VAD segments in streaming
NORMALIZE_AUDIO = os.getenv("NORMALIZE_AUDIO", "1") in {"1", "true", "True"}
TARGET_RMS = float(os.getenv("TARGET_RMS", "0.03"))  # ~-30 dBFS linear RMS
ADAPTIVE_VAD = os.getenv("ADAPTIVE_VAD", "0") in {"1", "true", "True"}
OUTPUT_JSONL = os.getenv("OUTPUT_JSONL", "")  # path to write JSONL transcripts

# VAD
# VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.6"))
# VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "250"))
# VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "250"))
# VAD_SPEECH_PAD_MS = int(os.getenv("VAD_SPEECH_PAD_MS", "150"))
# VAD_MAX_SPEECH_S = float(os.getenv("VAD_MAX_SPEECH_S", "20"))

# VAD tuned for continuous speech
VAD_THRESHOLD = 0.5          # Lower → more sensitive, catches quieter speech
VAD_MIN_SPEECH_MS = 500      # Require at least 0.5s of speech
# VAD_MIN_SILENCE_MS = 600     # Need ~0.6s silence before splitting segments
VAD_MIN_SILENCE_MS = 800  # or even 1000ms
VAD_SPEECH_PAD_MS = 200      # Add 0.2s padding before/after speech
VAD_MAX_SPEECH_S = 30.0      # Allow up to 30s per segment


# Transcription
# BEAM_SIZE = int(os.getenv("BEAM_SIZE", "5"))
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "1"))



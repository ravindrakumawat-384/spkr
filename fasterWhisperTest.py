from faster_whisper import WhisperModel

model_size = "small.en"

model = WhisperModel(model_size, device="cuda", compute_type="float16")

segments, info = model.transcribe("audio.mp3", language="en" beam_size=5)

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    print("segment.text: ", segment.text)

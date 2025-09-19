#!/usr/bin/env python3
"""Thin entrypoint that runs the modular STT pipeline in stt/."""

import argparse
import logging
from stt.config import MODEL_PATH
from stt.transcriber import VADTranscriber


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Realtime/file STT with faster-whisper + Silero VAD")
    parser.add_argument("--file", type=str, default="", help="Path to media file (wav/mp3/webm) to transcribe")
    parser.add_argument("--stream", action="store_true", help="Stream output per VAD segment for files")
    args = parser.parse_args()

    trans = VADTranscriber(MODEL_PATH)
    if args.file:
        if args.stream:
            trans.run_file_streaming(args.file)
        else:
            trans.run_file(args.file)
    else:
        trans.run()


if __name__ == "__main__":
    main()

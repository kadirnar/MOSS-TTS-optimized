"""Write each PCM chunk immediately, without buffering the complete utterance."""

import argparse
from contextlib import closing

from moss_tts import MossTTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Hello, this is streaming speech.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", default="speech.pcm")
    args = parser.parse_args()
    with MossTTS.from_pretrained() as tts:
        voice = tts.clone_voice(args.reference)
        with (
            closing(tts.stream(args.text, voice=voice)) as chunks,
            open(args.output, "wb") as output,
        ):
            for chunk in chunks:
                output.write(chunk.pcm16())
                output.flush()
                if chunk.frame == 0:
                    print(f"First audio: {chunk.elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()

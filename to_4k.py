import ffmpeg
import sys
import os

def convert_to_4k(input_file, output_file):
    try:
        # Ensure the input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file '{input_file}' does not exist.")

        # Execute FFmpeg to upscale to 4K
        (
            ffmpeg
            .input(input_file)
            .output(output_file, vf="scale=3840:2160", vcodec="libx264", crf=23, preset="medium")
            .run()
        )
        print(f"4K version saved as '{output_file}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_to_4k.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    convert_to_4k(input_file, output_file)

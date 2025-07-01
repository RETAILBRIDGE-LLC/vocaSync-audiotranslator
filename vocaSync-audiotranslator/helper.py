import gradio as gr
import boto3
import os
import json
import moviepy
import tempfile
import shutil

# Load config
with open("config.json") as f:
    config = json.load(f)

BUCKET = config["bucket"]
INPUT_PREFIX = config["input_prefix"]
OUTPUT_PREFIX = config["output_prefix"]
AWS_ACCESS_KEY = config["aws_access_key"]
AWS_SECRET_KEY = config["aws_secret_key"]

# S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Main function
def extract_audio(video_file_path, audio_format):
    try:
        if video_file_path is None:
            return "No file uploaded", None

        # Use original filename from path
        original_filename = os.path.basename(video_file_path)
        video_ext = os.path.splitext(original_filename)[1]

        # Copy to temp dir with original filename
        temp_video_path = os.path.join(tempfile.gettempdir(), original_filename)
        shutil.copy(video_file_path, temp_video_path)
        print(f"Video copied to: {temp_video_path}")

        # Upload video to S3
        s3_input_key = INPUT_PREFIX + original_filename
        try:
            s3.upload_file(temp_video_path, BUCKET, s3_input_key)
        except Exception as e:
            return f"S3 upload error (video): {e}", None

        # Set audio path
        audio_filename = os.path.splitext(original_filename)[0] + f".{audio_format}"
        audio_path = os.path.join(tempfile.gettempdir(), audio_filename)

        # Extract audio
        try:
            clip = moviepy.VideoFileClip(temp_video_path)
            clip.audio.write_audiofile(audio_path)
            clip.close()
        except Exception as e:
            return f"Audio extraction error: {e}", None

        # Upload audio to S3
        s3_output_key = OUTPUT_PREFIX + audio_filename
        try:
            s3.upload_file(audio_path, BUCKET, s3_output_key)
        except Exception as e:
            return f"S3 upload error (audio): {e}", None

        return "✅ Audio extracted and uploaded successfully!", audio_path

    except Exception as final_error:
        return f"Unexpected error: {final_error}", None



# Gradio Interface
interface = gr.Interface(
    fn=extract_audio,
    inputs=[
        gr.File(label="Upload MP4 Video", type="filepath"),  # ✅ fixed type
        gr.Dropdown(["mp3", "wav"], label="Select Audio Format")
    ],
    outputs=[
        gr.Text(label="Status"),
        gr.Audio(label="🔊 Preview / ⬇ Download", type="filepath")
    ],
    title="🎧 Audio Extractor from MP4",
    description="Upload an MP4 video, choose audio format, and download the extracted audio. Original filenames are preserved and files are uploaded to S3."
)

if __name__ == "__main__":
    interface.launch()
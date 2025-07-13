import gradio as gr
import boto3
import os
import json
import tempfile
import shutil
import uuid
import torch
from TTS.api import TTS
from moviepy import VideoFileClip
import librosa
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
from dtw import dtw
from resemblyzer import VoiceEncoder, preprocess_wav

# --- Create Config File ---
config_data = {
    "bucket": "vocasync",
    "input_prefix": "Inputvideo-forAudioExtraction/",
    "output_prefix": "Outputaudio-Extracted/",
    "aws_access_key": "aws_access_key",
    "aws_secret_key": "aws_secret_key"
}

with open("config.json", "w") as f:
    json.dump(config_data, f, indent=4)

# --- Load Config ---
with open("config.json") as f:
    config = json.load(f)

BUCKET = config["bucket"]
INPUT_PREFIX = config["input_prefix"]
OUTPUT_PREFIX = config["output_prefix"]
AWS_ACCESS_KEY = config["aws_access_key"]
AWS_SECRET_ACCESS_KEY = config["aws_secret_key"]

# --- Setup S3 Client ---
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# --- Load TTS Model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)

# --- Directories ---
TEMP_DIR = "temp_combined"
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Global Settings for Similarity Analysis ---
SAMPLING_RATE = 16000  # Resample audio to this rate for consistency

# --- Helper: Preprocess Audio for Resemblyzer ---
def preprocess_audio_for_resemblyzer(audio_path_str: str):
    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path_str}")
    return preprocess_wav(audio_path)

# --- Helper: Get Speaker Embedding ---
def get_speaker_embedding(audio_path: str, encoder):
    try:
        wav = preprocess_audio_for_resemblyzer(audio_path)
        embedding = encoder.embed_utterance(wav)
        return embedding
    except FileNotFoundError:
        print(f"File not found: {audio_path}")
        return None
    except Exception as e:
        print(f"Error processing {audio_path} for embedding: {e}")
        return None

# --- Speaker Embedding Similarity ---
def calculate_speaker_similarity(original_audio_path: str, cloned_audio_path: str):
    try:
        encoder = VoiceEncoder(device=device)
        original_embedding = get_speaker_embedding(original_audio_path, encoder)
        cloned_embedding = get_speaker_embedding(cloned_audio_path, encoder)

        if original_embedding is None or cloned_embedding is None:
            return None

        similarity = 1 - cosine(original_embedding, cloned_embedding)
        return similarity
    except Exception as e:
        print(f"Error in Speaker Similarity calculation: {e}")
        print("Ensure ffmpeg is installed and in PATH if Resemblyzer has trouble loading audio.")
        print("Ensure audio files are long enough for Resemblyzer (e.g., > 1-2 seconds).")
        return None

# --- Main Processing Function ---
def process_video_clone_and_analyze(video_file, text_input):
    try:
        # Step 1: Save video locally
        original_filename = os.path.basename(video_file)
        local_video_path = os.path.join(tempfile.gettempdir(), original_filename)
        shutil.copy(video_file, local_video_path)

        # Step 2: Extract Audio
        clip = VideoFileClip(local_video_path)
        audio_filename = os.path.splitext(original_filename)[0] + ".wav"
        speaker_audio_path = os.path.join(TEMP_DIR, audio_filename)
        clip.audio.write_audiofile(speaker_audio_path)
        clip.close()

        # Step 3: Upload original audio to S3
        s3_input_key = f"{INPUT_PREFIX}{audio_filename}"
        s3.upload_file(speaker_audio_path, BUCKET, s3_input_key)

        # Step 4: Clone Voice using input text
        output_audio_filename = f"cloned_{uuid.uuid4()}.wav"
        output_audio_path = os.path.join(TEMP_DIR, output_audio_filename)
        tts_model.tts_to_file(text=text_input, speaker_wav=speaker_audio_path, language="en", file_path=output_audio_path)

        # Step 5: Upload cloned audio to S3
        s3_output_key = f"{OUTPUT_PREFIX}{output_audio_filename}"
        s3.upload_file(output_audio_path, BUCKET, s3_output_key)
        s3_url = f"s3://{BUCKET}/{s3_output_key}"

        # Step 6: Calculate Similarity Metrics
        similarity_score = calculate_speaker_similarity(speaker_audio_path, output_audio_path)

        # Prepare output message
        output_message = f"✅ config.json created successfully!\n"
        output_message += f"✅ Cloned audio uploaded to: {s3_url}\n\n"
        if similarity_score is not None:
            output_message += f"Speaker Similarity (Cosine): {similarity_score:.4f}\n"
            output_message += "(Higher is better, closer to 1.0 means more similar vocal characteristics)\n"
        else:
            output_message += "⚠ Could not calculate speaker similarity.\n"

        return output_audio_path, output_message

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# --- Gradio UI ---
ui = gr.Interface(
    fn=process_video_clone_and_analyze,
    inputs=[
        gr.File(label="🎥 Upload MP4 Video", type="filepath"),
        gr.Textbox(label="📝 Text to Synthesize", placeholder="Hello, this is my cloned voice.", lines=3)
    ],
    outputs=[
        gr.Audio(label="🔊 Cloned Audio Output", type="filepath"),
        gr.Textbox(label="S3 Upload Status & Voice Similarity Metrics")
    ],
    title="🎤 Video-to-Cloned Voice Generator with Similarity Analysis",
    description="Extracts voice from uploaded video, clones it, generates speech for given text, and compares the cloned voice to the original."
)

if __name__ == "__main__":
    ui.launch(share=True)
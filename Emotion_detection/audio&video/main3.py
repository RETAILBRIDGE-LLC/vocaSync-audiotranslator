import gradio as gr
import os
import torch
import librosa
import numpy as np
from moviepy import VideoFileClip
import tempfile
import time
import boto3
import shutil
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from deepface import DeepFace
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

# === S3 Configuration ===
config_data = {
    "bucket": "vocasync",
    "input_prefix": "Inputvideo-forAudioExtraction/",
    "output_prefix": "Outputaudio-Extracted/",
    "aws_access_key": "aws_access_key",#replace key
    "aws_secret_key": "aws_secret_key"#replace key
}

# S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=config_data['aws_access_key'],
    aws_secret_access_key=config_data['aws_secret_key']
)

# === Load Audio Emotion Model (once) ===
audio_model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_name)
audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(audio_model_name)
audio_model.eval()

# === Emotion Functions ===
def predict_audio_emotion(audio_array, sampling_rate):
    inputs = audio_processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = audio_model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    return audio_model.config.id2label[predicted_id].lower()

def predict_video_emotion(frame):
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list):
            return analysis[0]['dominant_emotion'].lower()
        return analysis['dominant_emotion'].lower()
    except Exception:
        return "error"

def fuse_emotions(audio_emotion, video_emotion):
    if audio_emotion == video_emotion:
        return audio_emotion
    if audio_emotion == "error":
        return video_emotion
    if video_emotion == "error":
        return audio_emotion
    return video_emotion

# === Main Video Processor ===
def process_video_in_memory(video_path):
    clip = VideoFileClip(video_path)
    duration = int(clip.duration)
    results = []

    for start in range(0, duration, 5):
        end = min(start + 5, duration)

        try:
            audio_clip = clip.audio.subclip(start, end)
            audio_array = audio_clip.to_soundarray(fps=16000)
            if audio_array.ndim == 2:
                audio_array = np.mean(audio_array, axis=1)
            audio_emotion = predict_audio_emotion(audio_array, sampling_rate=16000)
        except Exception:
            audio_emotion = "error"

        try:
            frame = clip.get_frame((start + end) / 2)
            video_emotion = predict_video_emotion(frame)
        except Exception:
            video_emotion = "error"

        final_emotion = fuse_emotions(audio_emotion, video_emotion)
        results.append(f"{str(timedelta(seconds=start))} - {str(timedelta(seconds=end))}    {final_emotion}")

    return results

# === Upload to S3 ===
def upload_to_s3(file_path, output_filename):
    s3_key = os.path.join(config_data['output_prefix'], output_filename)
    s3.upload_file(file_path, config_data['bucket'], s3_key)
    return f"s3://{config_data['bucket']}/{s3_key}"

# === Gradio Interface Function ===
def analyze_video(uploaded_video):
    try:
        start = time.time()

        temp_input_path = os.path.join(tempfile.gettempdir(), os.path.basename(uploaded_video))
        shutil.copy(uploaded_video, temp_input_path)

        results = process_video_in_memory(temp_input_path)

        output_filename = os.path.splitext(os.path.basename(temp_input_path))[0] + "_output.txt"
        temp_output_path = os.path.join(tempfile.gettempdir(), output_filename)
        with open(temp_output_path, "w") as f:
            f.write("\n".join(results))

        s3_path = upload_to_s3(temp_output_path, output_filename)

        end = time.time()
        minutes, seconds = divmod(int(end - start), 60)

        return f"""✅ Video analyzed and results uploaded!

🕒 Time taken: {minutes} min {seconds} sec  
📤 Output file path: {s3_path}

🧾 Sample:
{chr(10).join(results[:5])}..."""

    except Exception as e:
        return f"❌ Error processing video: {str(e)}"

# === Gradio UI ===
iface = gr.Interface(
    fn=analyze_video,
    inputs=gr.File(label="Upload Video File (.mp4)", type="filepath"),
    outputs="text",
    title="🎥 Multimodal Emotion Analyzer",
    description="Upload a video. The system will analyze emotions using both audio and facial expressions every 5 seconds."
)

iface.launch()
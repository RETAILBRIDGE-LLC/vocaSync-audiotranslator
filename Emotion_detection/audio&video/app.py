import os
import torch
import librosa
import numpy as np
from moviepy import VideoFileClip
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from deepface import DeepFace
from datetime import timedelta
import tempfile
import gradio as gr
import warnings

warnings.filterwarnings("ignore")

# Load audio model
audio_model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_name)
audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(audio_model_name)
audio_model.eval()

# Emotion prediction functions
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

# Main processing logic
def analyze_emotions(video_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_vid:
        temp_vid.write(video_file.read())
        temp_vid_path = temp_vid.name

    clip = VideoFileClip(temp_vid_path)
    duration = int(clip.duration)
    results = []

    for start in range(0, duration, 5):
        end = min(start + 5, duration)

        # Audio
        try:
            audio_clip = clip.audio.subclip(start, end)
            audio_array = audio_clip.to_soundarray(fps=16000)
            if audio_array.ndim == 2:
                audio_array = np.mean(audio_array, axis=1)
            audio_emotion = predict_audio_emotion(audio_array, 16000)
        except Exception:
            audio_emotion = "error"

        # Video
        try:
            mid_time = (start + end) / 2
            frame = clip.get_frame(mid_time)
            video_emotion = predict_video_emotion(frame)
        except Exception:
            video_emotion = "error"

        final_emotion = fuse_emotions(audio_emotion, video_emotion)
        results.append([
            str(timedelta(seconds=start)),
            str(timedelta(seconds=end)),
            audio_emotion,
            video_emotion,
            final_emotion
        ])

    return results

# Gradio interface
iface = gr.Interface(
    fn=analyze_emotions,
    inputs=gr.Video(label="Upload .mp4 Video", type="file"),
    outputs=gr.Dataframe(headers=["Start", "End", "Audio Emotion", "Video Emotion", "Final Emotion"]),
    title="🎬 Emotion Detection (Audio + Video)",
    description="Uploads a video, then analyzes audio and facial emotions in 5-second segments.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()

import gradio as gr
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import pickle
from keras.models import load_model
from pydub import AudioSegment
import tempfile
import os

# Load ERM model and label encoder
# erm_model = load_model("ERM.h5")
erm_model = load_model("/home/ec2-user/SageMaker/AUDIO/ERM.h5")


with open("/home/ec2-user/SageMaker/AUDIO/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load YAMNet and class labels
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
yamnet_classes_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
yamnet_classes = [line.split(',')[2].strip() for line in tf.io.gfile.GFile(yamnet_classes_path).readlines()[1:]]

# Feature extraction for ERM
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

# ERM-based emotion prediction
def predict_erm(audio_path):
    features = extract_features(audio_path)
    features = features.reshape(1, 1, -1)
    predictions = erm_model.predict(features)
    predicted_index = np.argmax(predictions)
    emotion = label_encoder.classes_[predicted_index]
    return f"🧠 Speech Emotion (ERM): {emotion.upper()}"

# YAMNet-based prediction
def predict_yamnet(audio_path):
    waveform, sr = librosa.load(audio_path, sr=16000)
    waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    scores, _, _ = yamnet_model(waveform_tensor)
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    top_idx = np.argmax(mean_scores)
    label = yamnet_classes[top_idx]
    confidence = round(mean_scores[top_idx], 3)
    return label, confidence

# Main logic to decide model
def predict_combined(audio_file):
    # Convert mp3 to wav if needed
    if audio_file.endswith(".mp3"):
        sound = AudioSegment.from_file(audio_file, format="mp3")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sound.export(tmp.name, format="wav")
        audio_path = tmp.name
    else:
        audio_path = audio_file

    try:
        # Run YAMNet to classify
        yamnet_label, score = predict_yamnet(audio_path)

        # Use ERM only if YAMNet detects speech
        if "speech" in yamnet_label.lower():
            result = predict_erm(audio_path)
        else:
            result = f"🔉 Detected Sound (YAMNet): {yamnet_label}"
            # result = f"🔉 Detected Sound (YAMNet): {yamnet_label} ({score})"

        # Clean up temp file
        if audio_file.endswith(".mp3"):
            os.remove(audio_path)

        return result

    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio interface
interface = gr.Interface(
    fn=predict_combined,
    inputs=gr.Audio(type="filepath", label="Upload Audio (.wav or .mp3)"),
    outputs=gr.Textbox(label="Prediction"),
    title="🎙 Hybrid Emotion/Sound Detection",
    description="Automatically detects speech or sound.\nIf it's speech → uses ERM model\nIf it's sound → uses YAMNet."
)

if __name__ == "__main__":
    interface.launch()
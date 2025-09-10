import gradio as gr
import os
import tempfile
import shutil
import uuid
import torch
import librosa
import numpy as np
import soundfile as sf
import gc
from pathlib import Path
from TTS.api import TTS
# Media processing
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment

# AI Models & Utils
import whisper
from deep_translator import GoogleTranslator
from scipy.spatial.distance import cosine
from resemblyzer import VoiceEncoder, preprocess_wav

# ==== XTTS GLOBAL FIX FOR PYTORCH SAFE UNPICKLING ====
from TTS.tts.configs.xtts_config import XttsConfig, XttsArgs
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig

torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    BaseDatasetConfig,
    XttsArgs
])
# =================================================================

# ==== GLOBAL CONFIG & MODEL LOADING ====
print("Setting up global configurations and loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
TEMP_DIR = "temp_outputs"
os.makedirs(TEMP_DIR, exist_ok=True)

print("Loading TTS models (this may take a moment)...")
tts_models = {
    "YourTTS": TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device),
    "XTTS": TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device),
}
print("TTS models loaded successfully.")

# ============================ HELPER FUNCTIONS ============================
def clear_gpu_memory(): gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
def preprocess_audio_for_resemblyzer(p): return preprocess_wav(Path(p))
def get_speaker_embedding(p, e):
    try: return e.embed_utterance(preprocess_audio_for_resemblyzer(p))
    except Exception as err: print(f"Embedding Error: {err}"); return None

def calculate_speaker_similarity(original_audio_path: str, cloned_audio_path: str):
    try:
        encoder = VoiceEncoder(device=device)
        emb1 = get_speaker_embedding(original_audio_path, encoder)
        emb2 = get_speaker_embedding(cloned_audio_path, encoder)
        if emb1 is None or emb2 is None:
            return None
        return 1 - cosine(emb1, emb2)
    except Exception as e:
        print("Similarity Error:", e)
        return None
def process_simple_cloning(
    video_file,
    reference_audio,
    source_language,
    target_language,
    whisper_model_name,
    tts_model_name,
    temp_dir="temp_segments",
    emotion_temp_dir="temp_emotion_segments"
):
    # Save video
    original_filename = os.path.basename(video_file)
    local_video_path = os.path.join(tempfile.gettempdir(), original_filename)
    shutil.copy(video_file, local_video_path)

    # Extract Audio
    clip = VideoFileClip(local_video_path)
    audio_filename = os.path.splitext(original_filename)[0] + ".wav"
    source_audio = os.path.join(TEMP_DIR, audio_filename)
    clip.audio.write_audiofile(source_audio, logger=None)
    clip.close()

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(emotion_temp_dir, exist_ok=True)
    whisper_model = whisper.load_model(whisper_model_name)
    tts_model = tts_models[tts_model_name]

    # 1. Transcribe with timestamps
    transcription = whisper_model.transcribe(source_audio, language=source_language, fp16=False)
    segments = transcription["segments"]
    final_translated_text = ''
    sr = 16000
    final_audio = np.array([], dtype=np.float32)
    emotion_final_audio = np.array([], dtype=np.float32)
    current_time = 0.0
    emotion_current_time = 0.0

    for seg in segments:
        start, end, text = seg["start"], seg["end"], seg["text"]

        # --- Insert silence for pauses before this segment ---
        if start > current_time:
            silence_duration = start - current_time
            silence = np.zeros(int(silence_duration * sr), dtype=np.float32)
            final_audio = np.concatenate([final_audio, silence])

        if reference_audio and start > emotion_current_time:
            silence_duration = start - emotion_current_time
            silence = np.zeros(int(silence_duration * sr), dtype=np.float32)
            emotion_final_audio = np.concatenate([emotion_final_audio, silence])

        # --- Translate ---
        translated_text = GoogleTranslator(source="auto", target=target_language).translate(text)
        final_translated_text += translated_text + ' '

        # --- TTS (original speaker clone) ---
        out_wav = os.path.join(temp_dir, f"seg_{seg['id']}.wav")
        tts_model.tts_to_file(
            text=translated_text,
            file_path=out_wav,
            speaker_wav=source_audio,
            language=target_language,
        )
        speech, _ = librosa.load(out_wav, sr=sr)
        final_audio = np.concatenate([final_audio, speech])

        # --- TTS (emotion/reference speaker clone, only if provided) ---
        if reference_audio:
            emotion_out_wav = os.path.join(emotion_temp_dir, f"seg_{seg['id']}.wav")
            tts_model.tts_to_file(
                text=translated_text,
                file_path=emotion_out_wav,
                speaker_wav=reference_audio,
                language=target_language,
            )
            emotion_speech, _ = librosa.load(emotion_out_wav, sr=sr)
            emotion_final_audio = np.concatenate([emotion_final_audio, emotion_speech])
            emotion_current_time = end

        current_time = end

    # 3. Save final dubs
    final_audio_path = os.path.join(temp_dir, "final_dub.wav")
    sf.write(final_audio_path, final_audio, sr)

    similarity_score = calculate_speaker_similarity(source_audio, final_audio_path)
    similarity_text = f"{similarity_score:.4f}" if similarity_score else "⚠ Similarity Error"

    if reference_audio:
        emotion_final_audio_path = os.path.join(emotion_temp_dir, "emotion_final_dub.wav")
        sf.write(emotion_final_audio_path, emotion_final_audio, sr)
        emotion_similarity_score = calculate_speaker_similarity(source_audio, emotion_final_audio_path)
        emotion_similarity_text = f"{emotion_similarity_score:.4f}" if emotion_similarity_score else "⚠ Similarity Error"
    else:
        emotion_final_audio_path = None
        emotion_similarity_text = "(No reference audio given)"

    return (
        transcription["text"],
        final_translated_text,
        os.path.abspath(final_audio_path),
        os.path.abspath(emotion_final_audio_path) if reference_audio else None,
        similarity_text,
        emotion_similarity_text,
    )


# ============================ GRADIO UI (VERSION-COMPATIBLE & SIMPLIFIED) ============================
with gr.Blocks(theme=gr.themes.Soft(), title="VocaSync Suite") as ui:

    with gr.Tabs():
        with gr.TabItem("🎤 Voice & Emotion Cloning"):

            # ---------- Row 1 ----------

            with gr.Row():
                with gr.Column(scale=1):

                    Video_input = gr.File(label="🎥 Upload MP4 Video", type="filepath",scale=2)
                    with gr.Row():
                        source_language = gr.Dropdown(
                            label="Source Language",
                            choices=['hi','en','te','ta','ur'],
                            value='hi'
                        )

                        target_language = gr.Dropdown(
                            label="Target Language",
                            choices=['en'],
                            value='en'
                        )
                    with gr.Row():
                       whisper_model_simple = gr.Dropdown(
                    label="Whisper Model",
                    choices=["tiny", "base", "small", "medium"],
                    value="base"
                )

                       tts_model_simple = gr.Dropdown(
                    label="TTS Model",
                    choices=["YourTTS", "XTTS"],
                    value="XTTS"
                )



                with gr.Column(scale=1):
                    emotion_audio_input = gr.Audio(label="🎬 Reference emotion audio", type="filepath")


                    with gr.Row():
                      clone_btn_simple = gr.Button("🚀 Generate Dub", variant="primary")

            # ---------- Row  ----------
            with gr.Row():
                transcribed_text = gr.Textbox(
                    label="📝 Transcribed Text", interactive=False, lines=4
                )
                translated_text= gr.Textbox(
                    label="Translated Text", interactive=False, lines=4
                )

            with gr.Row():
                final_dub_output = gr.Audio(label="🎧 Output Audio")

                emotion_final_dub_output = gr.Audio(label="🎧 Reference Emotion Output Audio")
                with gr.Column(scale=1):
                  similarity_score = gr.Textbox(label="Similarity Score")
                  emotion_similarity_score = gr.Textbox(label="Reference emotion similarity Score")
            clone_btn_simple.click(
                          fn=process_simple_cloning,
                          inputs=[Video_input, emotion_audio_input, source_language, target_language, whisper_model_simple, tts_model_simple],
                          outputs=[transcribed_text, translated_text, final_dub_output, emotion_final_dub_output, similarity_score,emotion_similarity_score]
                      )

if __name__ == "__main__":
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    ui.launch(server_name="0.0.0.0", server_port=7860)
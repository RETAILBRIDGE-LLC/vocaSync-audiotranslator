import gradio as gr
import os
import tempfile
import shutil
import uuid
import torch  # Commented out - no GPU needed
from pathlib import Path
from moviepy.editor import VideoFileClip  # Commented out - no video processing
from scipy.spatial.distance import cosine  # Commented out - no similarity calculation
from resemblyzer import VoiceEncoder, preprocess_wav  # Commented out - no voice processing
import whisper  # Commented out - no transcription
from deep_translator import GoogleTranslator  # Commented out - no translation

# ==== Commented out all model-related code ====
# # ==== XTTS GLOBAL FIX FOR PYTORCH SAFE UNPICKLING ====
from TTS.tts.configs.xtts_config import XttsConfig, XttsArgs
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig

torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    BaseDatasetConfig,
    XttsArgs
])

# # ==== XTTS MODEL LOAD ====
from TTS.api import TTS
xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2",progress_bar=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
xtts_model.to(device)
yourtts_model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)

# ============================ TEMP DIR ============================
TEMP_DIR = "temp_combined"
os.makedirs(TEMP_DIR, exist_ok=True)
tts_models = {
    "YourTTS": yourtts_model,
    "XTTS": xtts_model,
}

# ============================ MOCK FUNCTIONS ============================
def preprocess_audio_for_resemblyzer(audio_path_str: str):
    audio_path = Path(audio_path_str)
    return preprocess_wav(audio_path)
    
def get_speaker_embedding(audio_path: str, encoder=None):
  try:
        wav = preprocess_audio_for_resemblyzer(audio_path)
        return encoder.embed_utterance(wav)
  except Exception as e:
        print("Embedding Error:", e)
        return None
  
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


def process_video_clone_and_analyze(video_file, target_language='en', model_choice='YourTTS'):
  try:
        whisper_model = whisper.load_model("small")

        # Save video`
        video_file_path = video_file.name 
        original_filename = os.path.basename(video_file_path)
        local_video_path = os.path.join(tempfile.gettempdir(), original_filename)
        shutil.copy(video_file_path, local_video_path)

        # Extract Audio
        clip = VideoFileClip(local_video_path)
        audio_filename = os.path.splitext(original_filename)[0] + ".wav"
        speaker_audio_path = os.path.join(TEMP_DIR, audio_filename)
        clip.audio.write_audiofile(speaker_audio_path, logger=None)
        clip.close()

        # Transcribe
        result = whisper_model.transcribe(speaker_audio_path)
        original_text = result["text"]

        # Translate
        translated_text = GoogleTranslator(source='auto', target=target_language).translate(original_text)

        # Clone Voice using XTTS
        output_audio_filename = f"cloned_{uuid.uuid4()}.wav"
        output_audio_path = os.path.join(TEMP_DIR, output_audio_filename)
        tts_model = tts_models[model_choice]
        tts_model.tts_to_file(
            text=translated_text,
            speaker_wav=speaker_audio_path,
            language=target_language,
            file_path=output_audio_path
        )

        # Compare Speakers
        similarity_score = calculate_speaker_similarity(speaker_audio_path, output_audio_path)
        similarity_text = f"{similarity_score:.4f} (Closer to 1.0 means more similar voice characteristics)" if similarity_score else "⚠ Could not compute speaker similarity."

        return speaker_audio_path, output_audio_path, original_text, translated_text, similarity_text
  except Exception as e:
        return None,None, "", "", f"❌ Error: {str(e)}"

  

# ============================ GRADIO UI ============================
with gr.Blocks(title="🎤 Voice Cloning & Translation (UI Demo)") as ui:
    gr.Markdown("## 🎤 Voice Cloning & Translation from Video")


    with gr.Row():
      video_input = gr.File(label="🎥 Upload MP4 Video", type="filepath", file_types=[".mp4"])
      lang_dropdown = gr.Dropdown(
          label="Target Language",
          choices=["en", "es", "fr", "de", "it"],
          value="en",
          scale=1
      )
      model_dropdown = gr.Dropdown(
          label="Voice Cloning Model",
          choices=["YourTTS", "XTTS"],
          value="XTTS",
          scale=1
      )
      submit_btn = gr.Button("🚀 Submit", scale=1)


    with gr.Row():
        with gr.Column(scale=1):
            original_audio = gr.Audio(label="Original Audio", type="filepath")
            original_text = gr.Textbox(label="Original Text", lines=4)

        with gr.Column(scale=1):
            cloned_audio = gr.Audio(label="Cloned Audio Output", type="filepath")
            translated_text = gr.Textbox(label="Translated Text", lines=4)
            similarity_score = gr.Textbox(label="Speaker Similarity Score")


    submit_btn.click(
        fn=process_video_clone_and_analyze,
        inputs=[video_input, lang_dropdown, model_dropdown],
        outputs=[original_audio, cloned_audio, original_text, translated_text, similarity_score]
    )

# Launch with local network access
ui.launch(server_name="0.0.0.0", server_port=7860)
import os
import cv2
from deepface import DeepFace
from collections import Counter
import time

# === Configuration ===
input_video = "inputs/facial_expressions.mp4"
cascade_path = "inputs/haarcascade_frontalface_default.xml"

resize_scale = 0.5
process_every_n_frames = 2
interval_duration = 5  # in seconds

# Extract name from input video and set output text file name
video_name = os.path.splitext(os.path.basename(input_video))[0]  # e.g., emotions
emotion_log_filename = f"{video_name}_output.txt"
emotion_log_path = os.path.join("textfile_outputs", emotion_log_filename)


# Create output folder
os.makedirs("textfile_outputs", exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(cascade_path)

# Open video
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    print("⚠️ Error: Could not read FPS from video. Please check if the input video exists and is valid.")
    cap.release()
    exit()
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = total_frames / fps

# Open text file to save emotions
emotion_log = open(emotion_log_path, "w")


frame_count = 0
start_time = time.time()

current_interval_start = 0
current_interval_end = interval_duration
interval_emotions = []

detected_faces_count = 0  # Count of frames where emotion was detected
processed_frames = 0      # Frames processed (not skipped)

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # in seconds

    if timestamp > current_interval_end:
        if interval_emotions:
            most_common_emotion = Counter(interval_emotions).most_common(1)[0][0]
        else:
            most_common_emotion = "no_face_detected"

        start_str = format_time(current_interval_start)
        end_str = format_time(current_interval_end)
        emotion_log.write(f"{start_str} - {end_str} {most_common_emotion}\n")

        current_interval_start += interval_duration
        current_interval_end += interval_duration
        interval_emotions = []

    # Skip some frames
    if frame_count % process_every_n_frames != 0:
        continue

    processed_frames += 1

    # Resize and detect face
    small_frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        x = int(x / resize_scale)
        y = int(y / resize_scale)
        w = int(w / resize_scale)
        h = int(h / resize_scale)
        face = frame[y:y+h, x:x+w]
        try:
            result = DeepFace.analyze(
                face,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='skip'
            )
            emotion = result[0]['dominant_emotion']
            interval_emotions.append(emotion)
            detected_faces_count += 1
        except Exception:
            continue

# Log remaining interval
if interval_emotions:
    most_common_emotion = Counter(interval_emotions).most_common(1)[0][0]
    start_str = format_time(current_interval_start)
    end_str = format_time(min(video_duration, current_interval_end))
    emotion_log.write(f"{start_str} - {end_str} {most_common_emotion}\n")

# Cleanup
cap.release()
emotion_log.close()
duration = time.time() - start_time

# Display detection rate
if processed_frames > 0:
    detection_rate = (detected_faces_count / processed_frames) * 100
else:
    detection_rate = 0

print(f"✅ Done in {duration:.2f} seconds!")
print(f"📝 Emotions saved to: {emotion_log_path}")

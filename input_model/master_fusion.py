import cv2
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
import torch
import librosa
from transformers import pipeline
from deepface import DeepFace
from test_audeering import Wav2Small # Your peer's blueprint file!

# --- CONFIGURATION ---
AUDIO_FILE = "current_turn.wav"
SAMPLE_RATE = 16000

# Global state to control our background threads
is_recording = False
video_frames = []
audio_data = []

# 💡 THE FIX: Initialize the webcam on the MAIN thread so Apple allows it!
print("📸 Initializing webcam (Please click 'OK' if Mac asks for permission)...")
cap = cv2.VideoCapture(0)
# Give the camera a second to warm up and adjust exposure
time.sleep(1)

# --- 1. THREAD: VIDEO RECORDER ---
def record_video():
    global is_recording, video_frames, cap
    last_capture_time = 0
    
    while is_recording:
        ret, frame = cap.read()
        if ret:
            current_time = time.time()
            # Snap exactly 1 frame every second so we don't melt the CPU
            if current_time - last_capture_time >= 1.0:
                video_frames.append(frame.copy())
                last_capture_time = current_time
        time.sleep(0.1) 

# --- 2. THREAD: AUDIO RECORDER ---
def audio_callback(indata, frames, time, status):
    if is_recording:
        audio_data.append(indata.copy())

# --- 3. MODEL INITIALIZATION ---
print("🧠 Waking up the Multimodal AI Brain... (This will take 10-15 seconds)")

# A. Whisper (Speech to Text)
print("  -> Loading Whisper...")
stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small.en")

# B. RoBERTa (Text to Emotion)
print("  -> Loading RoBERTa Text Emotions...")
text_emotion_pipeline = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

# C. Audeering (Prosodic Emotion)
print("  -> Loading Audeering Prosodic Emotions...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
audeering_model = Wav2Small.from_pretrained('audeering/wav2small').to(device).eval()

print("\n" + "="*60)
print("✅ SYSTEM READY. Awaiting your turn.")
print("="*60 + "\n")

if __name__ == "__main__":
    # --- START RECORDING TURN ---
    input("🎤 Press [ENTER] to start your turn...")
    is_recording = True
    video_frames = []
    audio_data = []
    
    # Start the hidden webcam thread
    vt = threading.Thread(target=record_video)
    vt.daemon = True # 💡 THE MAGIC BULLET: This prevents the zombie process!
    vt.start()
    
    # Start the microphone
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
            print("\n🔴 Recording! Speak naturally.")
            print("🛑 Press [Ctrl+C] when you are finished talking...")
            
            # This loop keeps the main thread alive safely without blocking the camera
            while True:
                time.sleep(0.1) 
                
    except KeyboardInterrupt:
        # 💡 We catch your Ctrl+C and use it as a smooth "End Turn" button!
        print("\n\n✅ Recording stopped. Processing Turn... Please wait.")
        is_recording = False
        
    vt.join(timeout=2.0) # Wait a max of 2 seconds for the webcam to safely turn off
 
    # Save the audio chunk
    print("\nProcessing Turn... Please wait.")
    full_audio = np.concatenate(audio_data, axis=0)
    sf.write(AUDIO_FILE, full_audio, SAMPLE_RATE)
    
    # --- RUN THE CLASSIFIERS ---
    
    # 1. TEXT TRANSLATION (Whisper)
    transcription = stt_pipeline(AUDIO_FILE)["text"].strip()
    
    # 2. TEXT EMOTION (RoBERTa)
    text_results = text_emotion_pipeline(transcription)[0]
    top_3_text = [(res['label'], res['score']) for res in text_results[:3]]
    
    # 3. PROSODIC EMOTION (Audeering)
    signal = torch.from_numpy(librosa.load(AUDIO_FILE, sr=SAMPLE_RATE)[0])[None, :]
    with torch.no_grad():
        logits = audeering_model(signal.to(device))
    arousal, dominance, valence = logits[0, 0].item(), logits[0, 1].item(), logits[0, 2].item()
    
    # 4. FACIAL EMOTION (DeepFace)
    # We average the emotions across all the frames we snapped while you were talking
    sum_emotions = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
    valid_frames = 0
    
    for frame in video_frames:
        try:
            # DeepFace can process the raw cv2 array directly!
            res = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)[0]
            for emo, score in res['emotion'].items():
                sum_emotions[emo] += score
            valid_frames += 1
        except:
            continue
            
    if valid_frames > 0:
        avg_emotions = {emo: score / valid_frames for emo, score in sum_emotions.items()}
        top_face_emo = max(avg_emotions, key=avg_emotions.get)
    else:
        avg_emotions = sum_emotions
        top_face_emo = "No face detected"

    # Turn off the camera hardware
    cap.release()

    # --- FINAL OUTPUT ---
    print("\n" + "="*60)
    print("🧠 MULTIMODAL FUSION PROFILE")
    print("="*60)
    print(f"🗣️  User Said: '{transcription}'\n")
    
    print("📖 TEXT MODALITY (RoBERTa):")
    for emo, score in top_3_text:
        print(f"   - {emo.capitalize()}: {score:.2f}")
        
    print("\n🎵 AUDIO MODALITY (Audeering):")
    print(f"   - Arousal (Energy) : {arousal:.2f}")
    print(f"   - Valence (Mood)   : {valence:.2f}")
    print(f"   - Dominance        : {dominance:.2f}")
    
    print("\n🎭 VIDEO MODALITY (DeepFace - Averaged over turn):")
    print(f"   - Dominant: {top_face_emo.capitalize()}")
    if valid_frames > 0:
        sorted_face = sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True)
        for emo, score in sorted_face[:3]: # Print top 3 face emotions
             print(f"   - {emo.capitalize()}: {score:.2f}%")
    print("="*60 + "\n")

    # --- SAVE FRAMES FOR DEBUGGING ---
    import os
    os.makedirs("debug_frames", exist_ok=True)
    for i, frame in enumerate(video_frames):
        cv2.imwrite(f"debug_frames/face_second_{i+1}.jpg", frame)
    print(f"\n📸 Saved {len(video_frames)} snapshot frames to the 'debug_frames' folder!")
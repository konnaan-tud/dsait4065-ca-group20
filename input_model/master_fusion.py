import cv2
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
import torch
import librosa
import requests
import json
from transformers import pipeline
from deepface import DeepFace
from input_model.prosodic_modality.prosodic_abstraction import ProsodyEmotionPredictor
from input_model.prosodic_modality.test_audeering import Wav2Small 
from input_model.prosodic_modality.vad_mapping import VADEmotionMapper, load_vad_prototypes

# --- CONFIGURATION ---
AUDIO_FILE = "current_turn.wav"
SAMPLE_RATE = 16000
OLLAMA_URL = "http://localhost:11434/api/generate"

# Global state
is_recording = False
video_frames = []
audio_data = []

print("📸 Initializing webcam (Please click 'OK' if Mac asks for permission)...")
cap = cv2.VideoCapture(0)
time.sleep(1)

# --- 1. THREAD: VIDEO RECORDER ---
def record_video():
    global is_recording, video_frames, cap
    last_capture_time = 0
    while is_recording:
        ret, frame = cap.read()
        if ret:
            current_time = time.time()
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
print("  -> Loading Whisper...")
stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small.en")
print("  -> Loading RoBERTa Text Emotions...")
text_emotion_pipeline = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
print("  -> Loading Audeering Prosodic Emotions...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
audeering_model = Wav2Small.from_pretrained('audeering/wav2small').to(device).eval()

print("\n" + "="*60)
print("✅ SYSTEM READY. Awaiting your turn.")
print("="*60 + "\n")

if __name__ == "__main__":
    input("🎤 Press [ENTER] to start your turn...")
    is_recording = True
    video_frames = []
    audio_data = []
    prosodic_predictor = ProsodyEmotionPredictor(device=device)
    vad_mapper = VADEmotionMapper(
        prototypes=load_vad_prototypes("prosodic_modality/vad_mapping.csv"),
        weights=(1.0, 1.0, 1.0),
        temperature=0.25,
    )
    
    vt = threading.Thread(target=record_video)
    vt.daemon = True
    vt.start()
    
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
            print("\n🔴 Recording! Speak naturally.")
            print("🛑 Press [Ctrl+C] when you are finished talking...")
            while True:
                time.sleep(0.1) 
    except KeyboardInterrupt:
        print("\n\n✅ Recording stopped.")
        is_recording = False
        
    vt.join(timeout=2.0)
 
    print("\nProcessing Turn and Benchmarking Latency... Please wait.")
    full_audio = np.concatenate(audio_data, axis=0)
    sf.write(AUDIO_FILE, full_audio, SAMPLE_RATE)
    
    # --- RUN THE CLASSIFIERS (WITH TIMERS) ---
    
    # 1. TEXT TRANSLATION (Whisper)
    t0 = time.time()
    transcription = stt_pipeline(AUDIO_FILE)["text"].strip()
    time_whisper = time.time() - t0
    
    # 2. TEXT EMOTION (RoBERTa)
    t0 = time.time()
    text_results = text_emotion_pipeline(transcription)[0]
    top_3_text = [(res['label'], res['score']) for res in text_results[:3]]
    time_roberta = time.time() - t0
    
    # 3. PROSODIC EMOTION (Audeering)
    t0 = time.time()

    # For now keeping everything the same, as I need to leave soon. 
    # Commented line below is example of how to predict Ekman from prosody in one line.
    #
    # ekman_probs = prosodic_predictor.predict_ekman(audio_path=AUDIO_FILE, sample_rate=SAMPLE_RATE)
    
    signal = torch.from_numpy(librosa.load(AUDIO_FILE, sr=SAMPLE_RATE)[0])[None, :]
    with torch.no_grad():
        logits = audeering_model(signal.to(device))
    arousal, dominance, valence = logits[0, 0].item(), logits[0, 1].item(), logits[0, 2].item()
    ekman_probs = vad_mapper.predict_proba((valence, arousal, dominance))
    time_audeering = time.time() - t0
    
    # 4. FACIAL EMOTION (DeepFace)
    t0 = time.time()
    sum_emotions = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
    valid_frames = 0
    for frame in video_frames:
        try:
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
    time_deepface = time.time() - t0

    cap.release()

    # --- 5. THE LLM DIALOG MANAGER ---
    print("\n🧠 Sending profile to Llama 3...")
    
    # We inject the actual scores into the prompt so Llama knows exactly how you feel!
    system_prompt = f"""
    You are an empathetic conversational agent. The user just said: "{transcription}"
    
    Here is the user's hidden emotional profile:
    - Their face looks mostly: {top_face_emo}
    - Their text implies: {top_3_text[0][0]} and {top_3_text[1][0]}
    - Their voice energy (Arousal) is: {arousal:.2f}
    
    Respond naturally to the user in 2-3 sentences. Use this emotional context to be deeply empathetic.
    """
    
    payload = {"model": "llama3", "prompt": system_prompt, "stream": False}
    
    t0 = time.time()
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        agent_reply = response.json().get("response", "Error generating response.")
    except Exception as e:
        agent_reply = "Could not connect to local Ollama LLM."
    time_llm = time.time() - t0

    # --- FINAL OUTPUT ---
    print("\n" + "="*60)
    print("🤖 AGENT RESPONSE")
    print("="*60)
    print(f"🗣️  User Said: '{transcription}'")
    print(f"\n💬 Agent: {agent_reply}")

    print("📖 TEXT MODALITY (RoBERTa):")
    for emo, score in top_3_text:
        print(f"   - {emo.capitalize()}: {score:.2f}")
        
    print("\n🎵 AUDIO MODALITY (Audeering):")
    print(f"   - Arousal (Energy) : {arousal:.2f}")
    print(f"   - Valence (Mood)   : {valence:.2f}")
    print(f"   - Dominance        : {dominance:.2f}")
    print(f"   - Ekman Probabilities: {ekman_probs}")
    
    print("\n🎭 VIDEO MODALITY (DeepFace - Averaged over turn):")
    print(f"   - Dominant: {top_face_emo.capitalize()}")
    if valid_frames > 0:
        sorted_face = sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True)
        for emo, score in sorted_face[:3]: # Print top 3 face emotions
             print(f"   - {emo.capitalize()}: {score:.2f}%")
    print("="*60 + "\n")
    
    print("\n" + "="*60)
    print("⏱️ LATENCY BENCHMARKING REPORT")
    print("="*60)
    print(f"  - Whisper (Speech to Text) : {time_whisper:.2f} seconds")
    print(f"  - RoBERTa (Text Emotion)   : {time_roberta:.2f} seconds")
    print(f"  - Audeering (Audio Emotion): {time_audeering:.2f} seconds")
    print(f"  - DeepFace (Video Emotion) : {time_deepface:.2f} seconds ({valid_frames} frames processed)")
    print(f"  - Llama 3 (LLM Generation) : {time_llm:.2f} seconds")
    print(f"  -------------------------------------------")
    total_time = time_whisper + time_roberta + time_audeering + time_deepface + time_llm
    print(f"  - TOTAL PIPELINE LATENCY   : {total_time:.2f} seconds")
    print("="*60 + "\n")

    # --- SAVE FRAMES FOR DEBUGGING ---
    import os
    os.makedirs("debug_frames", exist_ok=True)
    for i, frame in enumerate(video_frames):
        cv2.imwrite(f"debug_frames/face_second_{i+1}.jpg", frame)
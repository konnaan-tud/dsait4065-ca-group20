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
from test_audeering import Wav2Small 
import os
import sys
from database import PromptDatabase

# --- CONFIGURATION ---
AUDIO_FILE = "current_turn.wav"
SAMPLE_RATE = 16000
OLLAMA_URL = "http://localhost:11434/api/generate"

# Global state
is_recording = False

def print_final_output(transcription, top_3_text, arousal, valence, dominance,
                       top_face_emo, avg_emotions, valid_frames, agent_reply):
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
    
    print("\n🎭 VIDEO MODALITY (DeepFace - Averaged over turn):")
    print(f"   - Dominant: {top_face_emo.capitalize()}")
    if valid_frames > 0:
        sorted_face = sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True)
        for emo, score in sorted_face[:3]: # Print top 3 face emotions
             print(f"   - {emo.capitalize()}: {score:.2f}%")

# --- 1. THREAD: VIDEO RECORDER ---
def record_video(frames, cap):
    global is_recording
    last_capture_time = 0
    while is_recording:
        ret, frame = cap.read()
        if ret:
            current_time = time.time()
            if current_time - last_capture_time >= 1.0:
                frames.append(frame.copy())
                last_capture_time = current_time
        time.sleep(0.1)

# --- 3. MODEL INITIALIZATION ---
def model_initialization():
    print("🧠 Waking up the Multimodal AI Brain... (This will take 10-15 seconds)")
    print("  -> Loading Whisper...")
    stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small.en")
    print("  -> Loading RoBERTa Text Emotions...")
    text_emotion_pipeline = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    print("  -> Loading Audeering Prosodic Emotions...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    audeering_model = Wav2Small.from_pretrained('audeering/wav2small').to(device).eval()
    return stt_pipeline, text_emotion_pipeline, audeering_model, device

def save_debug_frames(video_frames):
    os.makedirs("debug_frames", exist_ok=True)
    for i, frame in enumerate(video_frames):
        cv2.imwrite(f"debug_frames/face_second_{i+1}.jpg", frame)

def process_audio(audio_data):
    global is_recording

    # --- 2. THREAD: AUDIO RECORDER ---
    def audio_callback(indata, frames, time, status):
        if is_recording:
            audio_data.append(indata.copy())

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
            print("\n🔴 Recording! Speak naturally.")
            print("🛑 Press [Ctrl+C] when you are finished talking...")
            while True:
                time.sleep(0.1) 
    except KeyboardInterrupt:
        print("\n\n✅ Recording stopped.")
        is_recording = False
    print("\nProcessing Turn and Benchmarking Latency... Please wait.")
    full_audio = np.concatenate(audio_data, axis=0)
    sf.write(AUDIO_FILE, full_audio, SAMPLE_RATE)

def process_video_frames(video_frames, cap):
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

    cap.release()
    return top_face_emo, avg_emotions, valid_frames

def generate_angent_reply(transcription, helper_events, top_3_text, arousal, valence, dominance,
                         top_face_emo, avg_emotions):
    print("\n🧠 Sending profile to Llama 3...")

    past_context_lines = []
    for e in helper_events:
        emotions = e.get("emotions") or {}
        emotions_str = ", ".join(
            f"{k}: {v:.2f}" for k, v in emotions.items() if isinstance(v, (int, float))
        )
        past_context_lines.append(f'  - "{e["text"]}" (emotions: {emotions_str})')
    past_context = "\n".join(past_context_lines) if past_context_lines else "  (none)"
    print('\n📚 Past similar prompts with emotional context:\n' + past_context)
    # We inject the actual scores into the prompt so Llama knows exactly how you feel!
    system_prompt = f"""
    You are an empathetic conversational agent. The user just said: "{transcription}"

    Here is the user's hidden emotional profile:
    - Their face looks mostly: {top_face_emo}
    - Their text implies: {top_3_text[0][0]} and {top_3_text[1][0]}
    - Their voice energy (Arousal) is: {arousal:.2f}

    Previous similar things they said (with their emotional state at the time): {past_context}

    Respond naturally to the user in 2-3 sentences. Use this emotional context to be deeply empathetic.
    """
    
    payload = {"model": "llama3", "prompt": system_prompt, "stream": False}
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        agent_reply = response.json().get("response", "Error generating response.")
    except Exception as e:
        agent_reply = "Could not connect to local Ollama LLM."
    return agent_reply

if __name__ == "__main__":
    print("📸 Initializing webcam (Please click 'OK' if Mac asks for permission)...")
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    stt_pipeline, text_emotion_pipeline, audeering_model, device = model_initialization()
    db = PromptDatabase(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chroma_db'))
    
    while True:
        print("\n" + "="*60)
        print("✅ SYSTEM READY. Awaiting your turn.")
        print("="*60 + "\n")
        
        input("🎤 Press [ENTER] to start your turn...")
        
        is_recording = True
        video_frames = []
        audio_data = []
        
        vt = threading.Thread(target=record_video, args=(video_frames, cap,))
        vt.daemon = True
        vt.start()
        
        process_audio(audio_data)
            
        vt.join(timeout=2.0)
        
        # --- RUN THE CLASSIFIERS (WITH TIMERS) ---
        
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

        top_face_emo, avg_emotions, valid_frames = process_video_frames(video_frames, cap)
        
        helper_events = db.query(transcription, n_results=3)
        print("\n📚 Similar past prompts in Chroma:")
        for event in helper_events:
            print(f"  - {event[:40]}...")
        
        # --- 5. THE LLM DIALOG MANAGER ---
        agent_reply = generate_angent_reply(transcription, helper_events, top_3_text, arousal, valence, dominance,
                                        top_face_emo, avg_emotions)

        print_final_output(transcription, top_3_text, arousal, valence, dominance,
                        top_face_emo, avg_emotions, valid_frames, agent_reply)
        save_debug_frames(video_frames)

        # --- 6. STORE IN CHROMA ---
        emotions_record = {
            **{f"text_{label}": float(score) for label, score in top_3_text},
            "audio_arousal": arousal,
            "audio_valence": valence,
            "audio_dominance": dominance,
            "face_dominant": top_face_emo,
            **{f"face_{emo}": float(score) for emo, score in avg_emotions.items()},
        }
        db.add(transcription, emotions_record)
        print(f"💾 Turn stored in Chroma (id: {transcription[:40]}...)")
# --- 1. PATCH HUGGING FACE SECURITY FIRST ---
import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

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
import os
from datetime import datetime
from transformers import pipeline
from deepface import DeepFace
from test_audeering import Wav2Small 
import sys
from database import PromptDatabase

# --- CONFIGURATION ---
AUDIO_FILE = "current_turn.wav"
SAMPLE_RATE = 16000
# Changed to /chat to provide the skeleton for memory module
OLLAMA_URL = "http://localhost:11434/api/chat"

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
    last_capture_time = time.time()
    while is_recording:
        ret, frame = cap.read()
        if ret:
            current_time = time.time()
            if current_time - last_capture_time >= 1.0:
                frames.append(frame.copy())
                last_capture_time = current_time

# --- 3. MODEL INITIALIZATION ---
def model_initialization():
    print("🧠 Waking up the Multimodal AI Brain... (This will take 10-15 seconds)")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("  -> Loading Whisper...")
    stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small.en", device=device)
    print("  -> Loading DistilRoBERTa Text Emotions (7 Ekman)...")
    text_emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    print("  -> Loading Audeering Prosodic Emotions...")
    audeering_model = Wav2Small.from_pretrained('audeering/wav2small').to(device).eval()
    return stt_pipeline, text_emotion_pipeline, audeering_model, device

def save_debug_frames(video_frames, turn_counter):
    os.makedirs("debug_frames", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, frame in enumerate(video_frames):
        cv2.imwrite(f"debug_frames/turn_{turn_counter}_{timestamp}_face_second_{i+1}.jpg", frame)

def process_audio(audio_data):
    global is_recording

    # --- 2. THREAD: AUDIO RECORDER ---
    def audio_callback(indata, frames, time, status):
        if is_recording:
            audio_data.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
        print("\n🔴 Recording! Speak naturally.")
        input("🛑 Press [ENTER] when you are finished talking...\n")
        
    print("\n✅ Recording stopped.")
    is_recording = False

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

    return top_face_emo, avg_emotions, valid_frames

def generate_agent_reply(transcription, helper_events, top_3_text, arousal, valence, dominance,
                         top_face_emo, avg_emotions, chat_history):
    print("\n🧠 Sending profile to LLM...")

    past_context_lines = []
    for e in helper_events:
        emotions = e.get("emotions") or {}
        emotions_str = ", ".join(
            f"{k}: {v:.2f}" for k, v in emotions.items() if isinstance(v, (int, float))
        )
        past_context_lines.append(f'  - "{e["text"]}" (emotions: {emotions_str})')
    past_context = "\n".join(past_context_lines) if past_context_lines else "  (none)"

    print('\n📚 Past similar prompts with emotional context:\n' + past_context)
    
    # Pack everything securely into the USER message
    contextual_user_message = f"""
    [Hidden Context for Agent]
    - User Face: {top_face_emo}
    - User Text implies: {top_3_text[0][0]} and {top_3_text[1][0]}
    - User Voice energy (Arousal): {arousal:.2f}
    - Past Similar Events: {past_context}
    
    User Said: "{transcription}"
    """
    
    # 1. Append the packaged message as the USER
    chat_history.append({"role": "user", "content": contextual_user_message})
    
    payload = {
        "model": "llama3", 
        "messages": chat_history,
        "stream": False,
        "think": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        agent_reply = response.json().get("message", {}).get("content", "Error generating response.")
    except Exception as e:
        agent_reply = "Could not connect to local Ollama LLM."
        
    # 2. Append the actual agent reply as the ASSISTANT so it remembers the conversation
    if agent_reply != "Error generating response." and agent_reply != "Could not connect to local Ollama LLM.":
        chat_history.append({"role": "assistant", "content": agent_reply})
        
    return agent_reply


if __name__ == "__main__":
    print("📸 Initializing webcam (Please click 'OK' if Mac asks for permission)...")
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    stt_pipeline, text_emotion_pipeline, audeering_model, device = model_initialization()
    db = PromptDatabase(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chroma_db'))
    
    chat_history = [
        {
            "role": "system", 
            "content": """You are an empathetic conversational agent. Your goal is to establish "common ground" with the user. The user is going to tell you about an emotional event. Use the "explicit confirmation" strategy: acknowledge their feelings, and ask a gentle and simple clarification question to explore the event further. Keep your response strictly under 3 sentences. Also, your question should ask about the event/story to keep the narrative flowing (e.g. "What happened next?", "What did you say to her?", "How did she react when you said that?"). Be warm and conversational. Note: You will be provided with the user's emotional state for each turn. Use this to inform your empathy, but do not explicitly read the exact scores back to the user."""
        }
    ]
    
    turn_counter = 1
    
    print("\n" + "="*60)
    print("✅ SYSTEM READY. Awaiting your turn.")
    print("="*60 + "\n")
    
    while True:

        
        print("\n" + "-"*60)
        user_cmd = input(f"🟢 TURN {turn_counter} | Press [ENTER] to start speaking (or type 'q' to quit): ")
        if user_cmd.strip().lower() == 'q':
            print("\n👋 Ending conversation. Goodbye!")
            break
        
        is_recording = True
        video_frames = []
        audio_data = []
        
        vt = threading.Thread(target=record_video, args=(video_frames, cap,))
        vt.daemon = True
        vt.start()
        
        process_audio(audio_data)
        
        vt.join(timeout=2.0)

        if len(audio_data) == 0:
            print("⚠️ No audio detected. Try again.")
            continue
            
        # MOVED THIS HERE: Now it is completely safe from crashing!
        print("\nProcessing Turn... Please wait.")
        full_audio = np.concatenate(audio_data, axis=0)
        sf.write(AUDIO_FILE, full_audio, SAMPLE_RATE)
        
        # --- RUN THE CLASSIFIERS (WITH TIMERS) ---
        
        # 1. TEXT TRANSLATION (Whisper)
        t0 = time.time()
        transcription = stt_pipeline(AUDIO_FILE)["text"].strip()
        time_whisper = time.time() - t0
        
        if not transcription:
            print("⚠️ Whisper didn't hear any words. Try speaking louder.")
            continue

        
        # 2. TEXT EMOTION (RoBERTa)
        t0 = time.time()
        # Keep ALL 7 results for the database
        all_text_emotions = text_emotion_pipeline(transcription)[0] 
        # Keep only Top 3 for the LLM prompt and printing
        top_3_text = [(res['label'], res['score']) for res in all_text_emotions[:3]]
        time_roberta = time.time() - t0

        # 3. PROSODIC EMOTION (Audeering)
        t0 = time.time()
        signal = torch.from_numpy(librosa.load(AUDIO_FILE, sr=SAMPLE_RATE)[0])[None, :]
        with torch.no_grad():
            logits = audeering_model(signal.to(device))
        arousal, dominance, valence = logits[0, 0].item(), logits[0, 1].item(), logits[0, 2].item()
        time_audeering = time.time() - t0
        
        # 4. FACIAL EMOTION (DeepFace)
        t0 = time.time()

        top_face_emo, avg_emotions, valid_frames = process_video_frames(video_frames, cap)
        time_deepface = time.time() - t0
        
        # --- 5. RETRIEVE SIMILAR PAST PROMPTS FROM CHROMA ---
        t0 = time.time()
        try:
            helper_events = db.query(transcription, n_results=3)
        except Exception:
            helper_events = [] # Failsafe if the database is empty on turn 1
        time_db = time.time() - t0
        
        # --- 6. THE LLM DIALOG MANAGER ---
        t0 = time.time()
        agent_reply = generate_agent_reply(transcription, helper_events, top_3_text, arousal,
                                            valence, dominance, top_face_emo, avg_emotions, chat_history)
        time_llm = time.time() - t0

        print_final_output(transcription, top_3_text, arousal, valence, dominance,
                        top_face_emo, avg_emotions, valid_frames, agent_reply)
        save_debug_frames(video_frames, turn_counter)

        # --- 7. STORE IN CHROMA ---
        emotions_record = {
            **{f"text_{res['label']}": float(res['score']) for res in all_text_emotions},
            "audio_arousal": arousal,
            "audio_valence": valence,
            "audio_dominance": dominance,
            "face_dominant": top_face_emo,
            **{f"face_{emo}": float(score) for emo, score in avg_emotions.items()},
        }
        db.add(transcription, emotions_record)
        print(f"💾 Turn stored in Chroma (id: {transcription[:40]}...)")

        # --- PRINT LATENCY REPORT ---
        print("\n" + "="*60)
        print("⏱️ LATENCY BENCHMARKING REPORT")
        print("="*60)
        print(f"  - Whisper (Speech to Text) : {time_whisper:.2f} seconds")
        print(f"  - RoBERTa (Text Emotion)   : {time_roberta:.2f} seconds")
        print(f"  - Audeering (Audio Emotion): {time_audeering:.2f} seconds")
        print(f"  - DeepFace (Video Emotion) : {time_deepface:.2f} seconds ({valid_frames} frames processed)")
        print(f"  - ChromaDB (Memory Fetch)  : {time_db:.2f} seconds")
        print(f"  - LLM Generation           : {time_llm:.2f} seconds")
        print(f"  -------------------------------------------")
        total_time = time_whisper + time_roberta + time_audeering + time_deepface + time_db + time_llm
        print(f"  - TOTAL PIPELINE LATENCY   : {total_time:.2f} seconds")
        print("="*60 + "\n")
        
        turn_counter += 1
        
    cap.release()
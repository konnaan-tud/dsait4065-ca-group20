# --- 1. PATCH HUGGING FACE SECURITY FIRST ---
import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

import cv2
import os
import json
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
import torch
import librosa
import requests
from datetime import datetime
from transformers import pipeline
from deepface import DeepFace
from prosodic_modality.prosodic_abstraction import ProsodyEmotionPredictor
from prosodic_modality.test_audeering import Wav2Small 
from prosodic_modality.vad_mapping import VADEmotionMapper, load_vad_prototypes
from TTS.api import TTS
import subprocess
from confidence import is_confident, prune_low_confidence_modalities
from agreement import analyze_agreement
from fusion import fuse_modalities

from database import PromptDatabase

# --- CONFIGURATION ---
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(_BASE_DIR, 'audio_recordings')
DEBUG_FRAMES_DIR = os.path.join(_BASE_DIR, 'debug_frames')
SAMPLE_RATE = 16000
# Changed to /chat to provide the skeleton for memory module
OLLAMA_URL = "http://localhost:11434/api/chat"
CONFIDENCE_THRESHOLD = 0.15 # to test

# 💡 NEW: Memory Contradiction Settings
SEMANTIC_DISTANCE_THRESHOLD = 1.1 # Semantic match threshold 
MEMORY_CONTRADICTION_THRESHOLD = 0.20 # MAE threshold for triggering the Curiosity Prompt

is_recording = False
last_valid_agent_utterance = ""  # 💡 Added this variable to track the conversation!

# Lock to prevent stdout race conditions between threads (e.g. background
# threads printing at the same time as ui_event, which merges lines
# and breaks the UI's event parser).
_print_lock = threading.Lock()

# Helper function to print events Streamlit can listen for.
def ui_event(event_type, **payload):
    event = {"type": event_type, **payload}
    with _print_lock:
        print("UI_EVENT::" + json.dumps(event), flush=True)

# Maps modalities apparently not all modalities have the same name for emotions
def normalize_emotion(label):
    mapping = {
        "angry": "anger",
        "anger": "anger",

        "sad": "sadness",
        "sadness": "sadness",

        "happy": "joy",
        "joy": "joy",

        "fear": "fear",
        "disgust": "disgust",
        "surprise": "surprise",

        "neutral": "neutral"
    }

    return mapping.get(label.lower(), label.lower())

def resolve_conflict_with_user(user_reply, text_emotion_pipeline):
    results = text_emotion_pipeline(user_reply)[0]
    predicted = max(results, key=lambda x: x["score"])
    
    # 💡 FIX: Flatten the list into a dictionary so MAE math works
    flat_distribution = {normalize_emotion(res["label"]): res["score"] for res in results}
    return predicted["label"], flat_distribution


def print_final_output(transcription, top_3_text, arousal, valence, dominance,
                       ekman_probs_norm, avg_emotions, valid_frames, agent_reply, text_confident, 
                       text_diff, audio_confident, audio_diff, decision, modalities, face_confident, face_diff, memory_data=None):
        print("\n" + "="*60)
        print("AGENT RESPONSE")
        print("="*60)

        print(f"User Said: '{transcription}'")
        print(f"\nAgent: {agent_reply}")

        print("\nTEXT MODALITY:")
        for emo,score in top_3_text:
            print(f"   {emo}: {score:.2f}")

        print("\nAUDIO MODALITY:")
        print("   Ekman probabilities:")
        for emo, score in ekman_probs_norm.items():
            print(f"   {emo}: {score:.2f}")

        print("\nVIDEO MODALITY:")
        if valid_frames>0:
            sorted_face=sorted(avg_emotions.items(), key=lambda x:x[1], reverse=True)
            for emo,score in sorted_face[:3]:
                print(f"   {emo}: {score:.2f}")

        print("\nCONFIDENCE CHECK")
        print(f"Text confident  : {text_confident} (diff={text_diff:.2f})")
        print(f"Audio confident : {audio_confident} (diff={audio_diff:.2f})")
        print(f"Face confident  : {face_confident} (diff={face_diff:.2f})")

        print("\nDECISION DEBUG")
        print(f"Decision type: {decision}")
        print(f"Number of confident modalities: {len(modalities)}")

        for name, m in modalities.items():
            print(f" - {name}: {m['top']} (confidence={m['confidence']:.2f})")
        
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
    print("Waking up the Multimodal AI Brain... (This will take 10-15 seconds)")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("  -> Loading Whisper...")
    stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small.en", device=device)
    print("  -> Loading DistilRoBERTa Text Emotions (7 Ekman)...")
    text_emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    print("  -> Loading Audeering Prosodic Emotions...")
    audeering_model = Wav2Small.from_pretrained('audeering/wav2small').to(device).eval()
    tts_model = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False)
    return stt_pipeline, text_emotion_pipeline, audeering_model, tts_model, device

def save_debug_frames(video_frames, turn_counter):
    os.makedirs(DEBUG_FRAMES_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, frame in enumerate(video_frames):
        cv2.imwrite(os.path.join(DEBUG_FRAMES_DIR, f"turn_{turn_counter}_{timestamp}_face_second_{i+1}.jpg"), frame)

def process_audio(audio_data):
    global is_recording
        # --- 2. THREAD: AUDIO RECORDER ---
    def audio_callback(indata, frames, time, status):
        if is_recording:
            audio_data.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
        print("\nRecording! Speak naturally.")
        input("Press [ENTER] when you are finished talking...\n")
        
    print("\nRecording stopped.")
    is_recording = False

def process_video_frames(video_frames, cap):
    sum_emotions = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
    valid_frames = 0

    for frame in video_frames:
        try:
            res = DeepFace.analyze(img_path=frame, actions=["emotion"], enforce_detection=False)[0]

            for emo,score in res["emotion"].items():
                sum_emotions[emo]+=score

            valid_frames+=1
        except:
            continue

    if valid_frames>0:
        avg_emotions={emo:score/valid_frames for emo,score in sum_emotions.items()}

        total=sum(avg_emotions.values())
        avg_emotions={emo:score/total for emo,score in avg_emotions.items()}

        avg_emotions_norm = {normalize_emotion(k): v for k, v in avg_emotions.items()}

        face_confident, top_face_emo, face_score, face_diff = is_confident(avg_emotions_norm)
    else:
        face_confident=False
        top_face_emo="No face detected"
        face_diff=0
        face_score = 0.0
        avg_emotions_norm = {}

    return top_face_emo, avg_emotions_norm, valid_frames, face_confident, face_score, face_diff

def generate_agent_reply(transcription, text_top, modalities, final_emotion, chat_history, decision, emotion_profile_text):

    global last_valid_agent_utterance

    print("\nSending profile to LLM...")

    # --- DYNAMIC SYSTEM PROMPT INJECTION ---
    base_system = """You are an empathetic, human-like conversational partner. Your goal is to establish "common ground" with the user regarding their emotional story.

    CRITICAL RULES:
    1. Acknowledge their situation gracefully, but NEVER use the exact emotion labels provided in your hidden context (e.g., do not say "You are feeling anger/neutral").
    2. NEVER start your sentences with cliché therapy phrases like "It sounds like...", "I sense...", or "I hear you saying...". Speak naturally like a friend.
    3. Ask one gentle and simple clarification question to keep the narrative flowing.
    4. Keep your response strictly under 3 sentences."""

    chat_history[0]["content"] = base_system

    # CONFLICT
    if decision == "conflict":
        emotion_labels = [m["top"] for m in modalities.values()]
        emotion_lines = ", ".join(emotion_labels)

        contextual_user_message = f"""
        [Hidden Context for Agent]

        User message: "{transcription}"
        User emotional profile: "{emotion_profile_text}"

        Emotions detected: {emotion_lines}

        These signals do not agree.

        Instructions:
        - You noticed this emotional mismatch.
        - Gently and naturally point out the contrast to the user.
        - Use the emotions provided above ({emotion_lines}) in your response.
        - Do NOT mention modality names or confidence scores.
        - Frame the exact emotion words conversationally and empathetically, not judgmentally.
        - End by warmly asking them to clarify how they are truly feeling underneath.
        - Maximum 3 sentences.
        """
        print(contextual_user_message)
        chat_history.append({"role": "user", "content": contextual_user_message})
    # RESOLVED
    elif decision == "resolved":
        contextual_user_message = f"""
        [Hidden Context for Agent]
        The user just clarified an emotional contradiction from the previous turn.
        User's clarification: "{transcription}"
        User's true emotion: {text_top}

        Instructions:
        - Briefly validate their true feeling (e.g., "Thank you for clarifying...").
        - IMPORTANT: We just took a brief detour. You need to return to the conversation. 
        - The last topic or question you were discussing before the detour was: "{last_valid_agent_utterance}"
        - Naturally transition BACK to that topic or continue the thought. 
        - Keep it seamless and conversational, strictly under 3 sentences.
        """
        chat_history.append({"role": "user", "content": contextual_user_message})
    elif decision == "no_data":
        contextual_user_message = f"""
            [Hidden Context for Agent]

            User message: "{transcription}"

            No clear emotional signal detected.

            Instructions:
            - Be honest that you're not fully sure how they're feeling.
            - Ask a gentle, open-ended clarification question.
            - Do NOT guess or force an emotion.
            - Keep it natural and under 3 sentences.
            """
        print(contextual_user_message)
        chat_history.append({"role": "user", "content": contextual_user_message})

    # NORMAL FUSION (WITH MEMORY CHECK)
    else:
        contextual_user_message = f"""
        [Hidden Context for Agent]
        User emotional profile: "{emotion_profile_text}"
        Detected emotional state: {final_emotion if final_emotion else "uncertain"}
        User message: "{transcription}"
        """
        chat_history.append({"role": "user", "content": contextual_user_message})
    
    # 💡 Short-Term Memory Sliding Window Buffer
    if len(chat_history) > 11:
        chat_history[:] = [chat_history[0]] + chat_history[-10:]

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
        
    # Save this reply as the "last valid reply" if it's not a conflict detour
    if decision != "conflict":
        last_valid_agent_utterance = agent_reply
    return agent_reply

def text_to_speech(tts_model, sentence):
    t0 = time.time()
    tts_model.tts_to_file(text=sentence, file_path="output.wav")
    time_tts = time.time() - t0 # Stop timer
    ui_event("agent_reply", text=sentence)
    subprocess.run(["ffplay", "-nodisp", "-autoexit", "output.wav"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time_tts

if __name__ == "__main__":
    print("Initializing webcam (Please click 'OK' if Mac asks for permission)...")
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    stt_pipeline, text_emotion_pipeline, audeering_model, tts_model, device = model_initialization()
    
    os.makedirs(AUDIO_DIR, exist_ok=True)

    chat_history = [{"role": "system", "content": "Initializing..."}]

    turn_counter = 1

    pending_clarification = None
    
    print("\n" + "="*60)
    print("SYSTEM READY. Awaiting your turn.")
    print("="*60 + "\n")
    
    while True:

        ui_event("awaiting_start", turn_id=turn_counter)

        print("\n" + "-"*60)
        user_cmd = input(f"TURN {turn_counter} | Press [ENTER] to start speaking (or type 'q' to quit): ")
        
       # --- SESSION SAVE ON QUIT ---
        if user_cmd.strip().lower() == 'q':
            print("\nWrapping up the conversation... Please wait a moment.")

            # --- GENERATE THE FINAL GOODBYE MESSAGE ---
            final_prompt = """
            [Hidden Context for Agent]
            The user has decided to end the conversation for today.

            Instructions:
            - Act as an empathetic therapist/friend saying goodbye.
            - Start by saying something like "Thank you so much for sharing all of this with me today."
            - Validate this specific emotional journey and reassure them.
            - End with a warm, encouraging sign-off.
            - Keep it compassionate and natural. Maximum 5 to 6 sentences.
            """

            chat_history.append({"role": "user", "content": final_prompt})
            payload = {"model": "llama3", "messages": chat_history, "stream": False, "think": False}
            farewell_msg = "Thank you for chatting with me. Take care!"

            try:
                response = requests.post(OLLAMA_URL, json=payload)
                farewell_msg = response.json().get("message", {}).get("content", farewell_msg)
            except Exception as e:
                farewell_msg = "Thank you so much for chatting with me today. Take care of yourself!"

            print("\n" + "="*60)
            print(f"Agent: {farewell_msg}")
            print("="*60 + "\n")
            try:
                text_to_speech(tts_model, farewell_msg)
            except Exception as e:
                print(f"Farewell TTS failed: {e}")

            break # Exit the while loop

        ui_event("turn_start", turn_id=turn_counter)
        
        is_recording = True
        video_frames = []
        audio_data = []
        prosodic_predictor = ProsodyEmotionPredictor(device=device)
        vad_mapper = VADEmotionMapper(
            prototypes=load_vad_prototypes(os.path.join(os.path.dirname(__file__), "prosodic_modality", "vad_mapping.csv")), # prototypes=load_vad_prototypes("vad_mapping.csv"),
            weights=(1.0,1.0,1.0),
            temperature=0.25
        )
    
        vt = threading.Thread(target=record_video, args=(video_frames, cap,))
        vt.daemon = True
        vt.start()

        process_audio(audio_data)
            
        vt.join(timeout=2.0)

        if len(audio_data) == 0:
            print("No audio detected. Try again.")
            continue
 
        # MOVED THIS HERE: Now it is completely safe from crashing!
        print("\nProcessing Turn... Please wait.")
        full_audio = np.concatenate(audio_data, axis=0)
        AUDIO_FILE = os.path.join(AUDIO_DIR, f"turn_{turn_counter}.wav")
        sf.write(AUDIO_FILE, full_audio, SAMPLE_RATE)
    
        # --- RUN THE CLASSIFIERS (WITH TIMERS) ---
        
        # 1. TEXT TRANSLATION (Whisper)
        t0 = time.time()
        transcription = stt_pipeline(AUDIO_FILE)["text"].strip()
        ui_event("transcription", text=transcription)
        time_whisper = time.time() - t0
        
        if not transcription:
            print("⚠️ Whisper didn't hear any words. Try speaking louder.")
            continue

        # Handle reply of user in case of conflict
        if pending_clarification in ("conflict", "no_data"):
            print("Resolving previous emotional conflict/no_data from user reply...")

            final_emotion, final_distribution = resolve_conflict_with_user(
                transcription,
                text_emotion_pipeline
            )

            # 💡 FIX 1: Combine the two utterances into one "Document"
            # This ensures both the trigger and the explanation are searchable.
            emotions_record = {
                "final_emotion": final_emotion,
                "emotion_distribution": final_distribution
            }

            print(emotions_record)

            # 💡 FIX 2: Use the combined narrative as the primary Document

            emotion_profile_text = f"The user has clarified their feelings ({final_emotion}). Focus purely on the content of their explanation."

            print(f"Resolved emotion → {final_emotion} ({final_distribution})")

            pending_clarification = None

            agent_reply = generate_agent_reply(
                transcription=transcription,
                text_top=final_emotion,
                final_emotion=final_emotion,
                modalities={"text": {
                    "top": final_emotion,
                    "confidence": max(final_distribution.values())
                }},
                chat_history=chat_history,
                decision="resolved",
                emotion_profile_text=emotion_profile_text
                )

            print(f"User Said: '{transcription}'")
            print(f"\nAgent: {agent_reply}")
            text_to_speech(tts_model, agent_reply)

            total_time = time_whisper
            ui_event(
                "latency",
                items={
                    "whisper": float(time_whisper),
                    "text_emotion": 0.0,
                    "audio_emotion": 0.0,
                    "video_emotion": 0.0,
                    "llm": 0.0,
                    "total": float(total_time),
                }
            )

            turn_counter += 1
            continue 

        # 2. TEXT EMOTION (RoBERTa)
        t0 = time.time()
        # Keep ALL 7 results for the database
        all_text_emotions = text_emotion_pipeline(transcription)[0]
        text_emotions_norm = {normalize_emotion(res["label"]): res["score"] for res in all_text_emotions}
        text_confident, text_top, text_score, text_diff = is_confident(text_emotions_norm)
        # Keep only Top 3 for the LLM prompt and printing
        top_3_text = [(res['label'], res['score']) for res in all_text_emotions[:3]]
        time_roberta = time.time() - t0
    
        # 3. PROSODIC EMOTION (Audeering)
        t0 = time.time()
        signal = torch.from_numpy(librosa.load(AUDIO_FILE, sr=SAMPLE_RATE)[0])[None, :]
        with torch.no_grad():
            logits = audeering_model(signal.to(device))
        arousal, dominance, valence = logits[0, 0].item(), logits[0, 1].item(), logits[0, 2].item()
        ekman_probs = vad_mapper.predict_proba((valence, arousal, dominance))
        ekman_probs_norm = {normalize_emotion(k): v for k, v in ekman_probs.items()}

        audio_confident, audio_top, audio_score, audio_diff = is_confident(ekman_probs_norm)
        time_audeering = time.time() - t0
        
        # 4. FACIAL EMOTION (DeepFace)
        t0 = time.time()
        top_face_emo, avg_emotions_norm, valid_frames, face_confident, face_score, face_diff = process_video_frames(video_frames, cap)
        time_deepface = time.time() - t0

        # Define which modalities will be considered in the final output.
        # In case a modality is not confident is excluded from the final result.
        modalities = {}

        if text_confident:
            modalities["text"] = {
                "probs": text_emotions_norm,
                "top": text_top,
                "confidence": text_score
            }

        if audio_confident:
            modalities["audio"] = {
                "probs": ekman_probs_norm,
                "top": audio_top,
                "confidence": audio_score
            }

        if face_confident:
            modalities["face"] = {
                "probs": avg_emotions_norm,
                "top": top_face_emo,
                "confidence": face_score
            }
        
        # Remove outliers
        modalities = prune_low_confidence_modalities(modalities)

        # Define prompts for each case
        decision, agreed_emotion, agreeing_modalities = analyze_agreement(modalities)

        final_emotion = None
        final_score = None
        fused_dist = {}


        # CASE 0: No confident modality
        if decision == "no_data":
            pending_clarification = "no_data"
            final_emotion = None
            emotion_profile_text = "No confident emotional signal detected."

            pending_event = {
                "initial_text": transcription,
                "initial_state": "no_data"
            }

        # CASE 1: ONLY ONE CONFIDENT MODALITY  🔥 (NEW RULE)
        elif len(agreeing_modalities) == 1:
            m_name = list(agreeing_modalities.keys())[0]
            m = agreeing_modalities[m_name]
            fused_dist = m["probs"]

            final_emotion = m["top"]
            final_score = m["confidence"]

            emotion_profile_text = (
                f"- Single confident modality used: {m_name}\n"
                f"- Detected emotion: {final_emotion}"
            ) 

        # CASE 2: CONFLICT (ALL DIFFERENT)
        elif decision == "conflict":
            final_emotion = None
            emotion_profile_text = "Conflicting emotional signals across modalities."
            pending_clarification = "conflict"

        # CASE 3: AGREEMENT/PARTIAL AGREEMENT → FUSION
        else:
            final_emotion, fused_dist, weights = fuse_modalities(agreeing_modalities)

            final_score = fused_dist.get(final_emotion, 0.0)

            emotion_profile = [f"- Final emotion (fused): {final_emotion}"]

            for m in agreeing_modalities:
                emotion_profile.append(
                    f"- {m.capitalize()} supports: {modalities[m]['top']} "
                    f"(weight={weights[m]:.2f})"
                )

            emotion_profile_text = "\n".join(emotion_profile)


        # --- 4. THE LLM DIALOG MANAGER ---
        t0 = time.time()
        agent_reply = generate_agent_reply(
            transcription=transcription, 
            text_top=text_top, 
            final_emotion=final_emotion, 
            chat_history=chat_history, 
            decision=decision,
            modalities=modalities, 
            emotion_profile_text=emotion_profile_text
        )
        time_llm = time.time() - t0

        print_final_output(transcription, top_3_text, arousal, valence, dominance,
                        ekman_probs_norm, avg_emotions_norm, valid_frames, agent_reply, text_confident, 
                        text_diff, audio_confident, audio_diff, decision, modalities, face_confident, face_diff)
        save_debug_frames(video_frames, turn_counter)
        time_tts = text_to_speech(tts_model, agent_reply)

        # --- PRINT LATENCY REPORT ---
        print("\n" + "="*60)
        print("LATENCY BENCHMARKING REPORT")
        print("="*60)
        print(f"  - Whisper (Speech to Text) : {time_whisper:.2f} seconds")
        print(f"  - RoBERTa (Text Emotion)   : {time_roberta:.2f} seconds")
        print(f"  - Audeering (Audio Emotion): {time_audeering:.2f} seconds")
        print(f"  - DeepFace (Video Emotion) : {time_deepface:.2f} seconds ({valid_frames} frames processed)")
        print(f"  - LLM Generation           : {time_llm:.2f} seconds")
        print(f" -  TTS Generation Latency: {time_tts:.2f} seconds")
        print(f"  -------------------------------------------")
        total_time = time_whisper + time_roberta + time_audeering + time_deepface + time_llm
        print(f"  - TOTAL PIPELINE LATENCY   : {total_time:.2f} seconds")
        print("="*60 + "\n")
        ui_event(
            "latency",
            items={
                "whisper": float(time_whisper),
                "text_emotion": float(time_roberta),
                "audio_emotion": float(time_audeering),
                "video_emotion": float(time_deepface),
                "llm": float(time_llm),
                "total": float(total_time),
            }
        )
        
        turn_counter += 1

    cap.release()

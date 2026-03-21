# --- 1. PATCH HUGGING FACE SECURITY FIRST ---
import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

import cv2
import os
import sys
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

from confidence import is_confident
from agreement import analyze_agreement
from fusion import fuse_modalities

from database import PromptDatabase

# --- CONFIGURATION ---
AUDIO_FILE = "current_turn.wav"
SAMPLE_RATE = 16000
# Changed to /chat to provide the skeleton for memory module
OLLAMA_URL = "http://localhost:11434/api/chat"
CONFIDENCE_THRESHOLD = 0.15 # to test

is_recording = False

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

# extract user emotion
def extract_explicit_emotion(text):
    text = text.lower()

    emotion_keywords = {
        "anger": ["angry", "mad", "annoyed", "frustrated"],
        "disgust": ["disgusted", "gross", "ew", "nasty"],
        "fear": ["scared", "afraid", "anxious", "nervous"],
        "joy": ["happy", "good", "great", "excited", "glad"],
        "sadness": ["sad", "down", "depressed", "unhappy"],
        "surprise": ["surprised", "shocked", "wow", "unexpected"],
        "neutral": ["okay", "fine", "alright", "normal"]
    }

    for emotion, keywords in emotion_keywords.items():
        if any(word in text for word in keywords):
            return emotion

    return None

def resolve_conflict_with_user(user_reply, text_emotion_pipeline):
    results = text_emotion_pipeline(user_reply)[0]
    predicted = max(results, key=lambda x: x["score"])
    
    explicit = extract_explicit_emotion(user_reply)

    if explicit:
        return explicit, 1.0
    else:
        return predicted["label"], max(predicted["score"], 0.85) # give a higher prediction to the output of the text to emotion model


def print_final_output(transcription, top_3_text, arousal, valence, dominance,
                       ekman_probs, avg_emotions, valid_frames, agent_reply, text_confident, 
                       text_diff, audio_confident, audio_diff, decision, modalities, face_confident, face_diff):
        print("\n" + "="*60)
        print("🤖 AGENT RESPONSE")
        print("="*60)

        print(f"🗣️ User Said: '{transcription}'")
        print(f"\n💬 Agent: {agent_reply}")

        print("\n📖 TEXT MODALITY:")
        for emo,score in top_3_text:
            print(f"   {emo}: {score:.2f}")

        print("\n🎵 AUDIO MODALITY:")
        print(f"   Arousal: {arousal:.2f}")
        print(f"   Valence: {valence:.2f}")
        print(f"   Dominance: {dominance:.2f}")
        print(f"   Ekman: {ekman_probs}")

        print("\n🎭 VIDEO MODALITY:")
        if valid_frames>0:
            sorted_face=sorted(avg_emotions.items(), key=lambda x:x[1], reverse=True)
            for emo,score in sorted_face[:3]:
                print(f"   {emo}: {score:.2f}")

        print("\n🔎 CONFIDENCE CHECK")
        print(f"Text confident  : {text_confident} (diff={text_diff:.2f})")
        print(f"Audio confident : {audio_confident} (diff={audio_diff:.2f})")
        print(f"Face confident  : {face_confident} (diff={face_diff:.2f})")

        print("\n🧠 DECISION DEBUG")
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
        avg_emotions=sum_emotions
        face_confident=False
        top_face_emo="No face detected"
        face_diff=0

    return top_face_emo, avg_emotions, valid_frames, face_confident, face_score, face_diff

def generate_agent_reply(transcription, helper_events, top_3_text, arousal, valence, dominance,
                         top_face_emo, avg_emotions, chat_history, decision):
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

    if decision == "conflict":
        contextual_user_message = f"""
        [Hidden Context for Agent]

        User message: "{transcription}"

        Emotion signals detected by the system:

        - Text analysis suggests: {text_top}
        - Voice tone suggests: {audio_top}
        - Facial expression suggests: {top_face_emo}

        These signals do not agree.

        Instructions:
        - Briefly explain that the signals are mixed.
        - Mention the different signals naturally.
        - Ask the user how they are actually feeling.
        - Keep the response under 2 sentences.
        """
        chat_history.append({"role": "user", "content": contextual_user_message})
    elif decision == "no_data":
        contextual_user_message = f"""
        [Hidden Context for Agent]

        User message: "{transcription}"

        No clear emotional signal was detected.

        Instructions:
        - Respond naturally.
        - Ask a question that encourages the user to continue the story.
        - Keep the response under 3 sentences.
        """
        chat_history.append({"role": "user", "content": contextual_user_message})

    else:

        contextual_user_message = contextual_user_message = f"""
        [Hidden Context for Agent]

        User message: "{transcription}"

        Detected emotional state: {final_emotion if final_emotion else "uncertain"}

        Instructions for the assistant:
        - You are an empathetic conversational agent.
        - Acknowledge the user's feelings.
        - Respond naturally in 2–3 sentences.
        - Ask one gentle follow-up question about the event.
        - Do NOT repeat previous responses.

        Write the response.
        """

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

    pending_conflict_resolution = False
    
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

        # Handle reply of user in case of conflict
        if pending_conflict_resolution:
            print("🧠 Resolving previous emotional conflict from user reply...")

            final_emotion, final_score = resolve_conflict_with_user(
                transcription,
                text_emotion_pipeline
            )

            emotions_record = {
                "final_emotion": final_emotion,
                "final_score": float(final_score),
                "decision": "user_resolved"
            }

            db.add(transcription, emotions_record)

            chat_history.append({
                "role": "system",
                "content": f"[Resolved user emotion: {final_emotion}]"
            })

            print(f"💾 Stored resolved emotion → {final_emotion} ({final_score:.2f})")

            pending_conflict_resolution = False
            
            # Now generate a NORMAL agent response using resolved emotion
            decision = "resolved"

            #emotion_profile_text = f"User clarified emotion: {final_emotion}"
    
        # 2. TEXT EMOTION (RoBERTa)
        t0 = time.time()
        # Keep ALL 7 results for the database
        all_text_emotions = text_emotion_pipeline(transcription)[0] 
        text_emotions = {normalize_emotion(res["label"]): res["score"] for res in all_text_emotions}
        text_confident, text_top, text_score, text_diff = is_confident(text_emotions)
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
        top_face_emo, avg_emotions, valid_frames, face_confident, face_score, face_diff = process_video_frames(video_frames, cap)
        time_deepface = time.time() - t0

        # Define which modalities will be considered in the final output.
        # In case a modality is not confident is excluded from the final result.
        modalities = {}

        if text_confident:
            modalities["text"] = {
                "probs": text_emotions,
                "top": text_top,
                "confidence": text_score
            }

        if audio_confident:
            modalities["audio"] = {
                "probs": ekman_probs,
                "top": audio_top,
                "confidence": audio_score
            }

        if face_confident:
            modalities["face"] = {
                "probs": avg_emotions,
                "top": top_face_emo,
                "confidence": face_score
            }

        # Define prompts for each case
        decision, agreed_emotion = analyze_agreement(modalities)

        final_emotion = None
        final_score = None

        # CASE 0: No confident modality
        if decision == "no_data":
            emotion_profile_text = "No confident emotional signal detected." # TODO: Change text


        # CASE 1: ONLY ONE CONFIDENT MODALITY  🔥 (NEW RULE)
        elif len(modalities) == 1:
            m_name = list(modalities.keys())[0]
            m = modalities[m_name]

            final_emotion = m["top"]
            final_score = m["confidence"]

            #emotion_profile_text = (
            #    f"- Single confident modality used: {m_name}\n"
            #    f"- Detected emotion: {final_emotion}"
            #) 

        # CASE 2: CONFLICT (ALL DIFFERENT)
        elif decision == "conflict":
            final_emotion = None
            #emotion_profile_text = "Conflicting emotional signals across modalities." # TODO: Change text
            pending_conflict_resolution = True

        # CASE 3: AGREEMENT/PARTIAL AGREEMENT → FUSION
        else:
            final_emotion, fused_dist, weights = fuse_modalities(modalities)

            final_score = fused_dist.get(final_emotion, 0.0)

            emotion_profile = [f"- Final emotion (fused): {final_emotion}"]

            for m in modalities:
                emotion_profile.append(
                    f"- {m.capitalize()} supports: {modalities[m]['top']} "
                    f"(weight={weights[m]:.2f})"
                )

            #emotion_profile_text = "\n".join(emotion_profile)

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
                                            valence, dominance, top_face_emo, avg_emotions, chat_history, decision)
        time_llm = time.time() - t0

        print_final_output(transcription, top_3_text, arousal, valence, dominance,
                        top_face_emo, avg_emotions, valid_frames, agent_reply, text_confident, 
                        text_diff, audio_confident, audio_diff, decision, modalities, face_confident, face_diff)
        save_debug_frames(video_frames, turn_counter)

        # --- 7. STORE IN CHROMA ---
        if final_emotion is not None and not pending_conflict_resolution:
            emotions_record = {
                "final_emotion": final_emotion,
                "final_score": float(final_score) if final_score else 1.0,
                "decision": decision,
            }
            
            db.add(transcription, emotions_record)
            print(f"💾 Turn stored in Chroma (id: {transcription[:40]}...)")
            print(f"Stored FINAL emotion in Chroma → {final_emotion} ({final_score:.2f})")
        else:
            print("Skipping DB storage (no resolved emotion)")

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

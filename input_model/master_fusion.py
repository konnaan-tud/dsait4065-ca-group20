# --- 1. PATCH HUGGING FACE SECURITY FIRST ---
import transformers.utils.import_utils
import transformers.modeling_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

import cv2
import json
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
AUDIO_FILE = "current_turn.wav"
SAMPLE_RATE = 16000
# Changed to /chat to provide the skeleton for memory module
OLLAMA_URL = "http://localhost:11434/api/chat"
CONFIDENCE_THRESHOLD = 0.15 # to test

# 💡 NEW: Memory Contradiction Settings
SEMANTIC_DISTANCE_THRESHOLD = 1.1 # Semantic match threshold 
MEMORY_CONTRADICTION_THRESHOLD = 0.20 # MAE threshold for triggering the Curiosity Prompt

is_recording = False
last_valid_agent_utterance = ""  # 💡 Added this variable to track the conversation!

# --- HYBRID MEMORY STATE VARIABLES ---
narrative_summary = ""
summary_lock = threading.Lock()
turns_for_summary = [] # Buffer to hold the recent turns before summarizing

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

def fetch_temporal_memory(db, current_text, current_emotion_dist):
    """Fetches past events and calculates the MAE emotional contradiction."""
    memory_data = {"past_text": None, "mae": 0.0, "is_contradiction": False, "past_top": None}
    
    try:
        results = db.query(current_text, n_results=1)
        
        if results:
            dist = results[0].get("distance", 999.0)
            
            if dist < SEMANTIC_DISTANCE_THRESHOLD:
                past_event = results[0]
                p_text = past_event.get("text", "")
                
                # Safely unpack the nested dictionary
                p_dist = past_event.get("emotions", {}).get("emotion_distribution", {})
                p_top = past_event.get("emotions", {}).get("final_emotion", "unknown")
                
                if p_dist:
                    emotions_list = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
                    # Calculate Mean Absolute Error
                    mae = sum(abs(current_emotion_dist.get(e, 0.0) - p_dist.get(e, 0.0)) for e in emotions_list) / 7.0
                    
                    memory_data = {
                        "past_text": p_text,
                        "mae": mae,
                        "is_contradiction": mae > MEMORY_CONTRADICTION_THRESHOLD,
                        "past_top": p_top
                    }
    except Exception as e:
        pass

    return memory_data

def print_final_output(transcription, top_3_text, arousal, valence, dominance,
                       ekman_probs_norm, avg_emotions, valid_frames, agent_reply, text_confident, 
                       text_diff, audio_confident, audio_diff, decision, modalities, face_confident, face_diff, memory_data=None):
        print("\n" + "="*60)
        print("🤖 AGENT RESPONSE")
        print("="*60)

        print(f"🗣️ User Said: '{transcription}'")
        print(f"\n💬 Agent: {agent_reply}")

        print("\n📖 TEXT MODALITY:")
        for emo,score in top_3_text:
            print(f"   {emo}: {score:.2f}")

        print("\n🎵 AUDIO MODALITY:")
        print("   Ekman probabilities:")
        for emo, score in ekman_probs_norm.items():
            print(f"   {emo}: {score:.2f}")

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
        
        # 💡 NEW: Cleanly print the Memory Match right here in the final output!
        if memory_data and memory_data.get("past_text"):
            print("\n🕰️ TEMPORAL MEMORY CHECK")
            print(f"Past Event Matched: '{memory_data['past_text']}'")
            print(f"Past Dominant Emo : {memory_data['past_top']}")
            print(f"Vector Distance   : {memory_data['mae']:.3f} (MAE)")
            if memory_data['is_contradiction']:
                print(f"   ⚠️ Memory vs Present Events Contradiction Detected (Threshold > {MEMORY_CONTRADICTION_THRESHOLD})")
            else:
                print(f"   ✅ Memory & Present Events Alignment Detected (Threshold <= {MEMORY_CONTRADICTION_THRESHOLD})")

# --- HYBRID MEMORY: ASYNCHRONOUS RUNNING SUMMARY ---
def update_running_summary(recent_turns, current_summary):
    global narrative_summary
    print("\n🔄 [Semantic Memory] Background thread summarizing recent turns...")
    
    transcript = "\n".join(recent_turns)
    
    prompt = f"""
    You are managing the Semantic Memory for an empathetic AI agent. 
    Update the user's psychological profile based on the new dialogue.

    You MUST structure your response EXACTLY in these two bulleted sections:
    1. Core Facts & Context: (Preserve specific details, nouns, and events mentioned by the user. Add new facts without deleting important old ones. Maximum 4 bullet points).
    2. Emotional Trajectory: (Analyze how their mood or core struggle is shifting right now. Maximum 2 bullet points).
    
    IMPORTANT: Output ONLY the two bulleted sections. Do not include introductory phrases like "Here is the summary".
    
    Previous Profile:
    {current_summary if current_summary else "None (Beginning of conversation)"}
    
    New Dialogue:
    {transcript}
    """

    payload = {
        "model": "llama3", 
        "messages": [{"role": "system", "content": prompt}],
        "stream": False,
        "think": False
    }
    
    try:
        t0_summary = time.time()
        response = requests.post(OLLAMA_URL, json=payload)
        time_summary = time.time() - t0_summary
        
        new_summary = response.json().get("message", {}).get("content", "").strip()
        if new_summary:
            with summary_lock:
                narrative_summary = new_summary
            print(f"\n✅ [Semantic Memory] Running Summary Updated in Background! (Latency: {time_summary:.2f} seconds)")
    except Exception as e:
        print(f"\n⚠️ [Semantic Memory] Summary update failed: {e}")

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
    tts_model = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False)
    return stt_pipeline, text_emotion_pipeline, audeering_model, tts_model, device

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
        face_confident=False
        top_face_emo="No face detected"
        face_diff=0
        face_score = 0.0
        avg_emotions_norm = {}

    return top_face_emo, avg_emotions_norm, valid_frames, face_confident, face_score, face_diff

def generate_agent_reply(transcription, text_top, modalities,
                         final_emotion, chat_history, decision, emotion_profile_text, memory_data=None):

    global last_valid_agent_utterance

    print("\n🧠 Sending profile to LLM...")

    with summary_lock:
        current_summary = narrative_summary

    if current_summary:
        print(f'\n🧠 Current Semantic Summary:\n  {current_summary}')

    # --- DYNAMIC SYSTEM PROMPT INJECTION ---
    base_system = """You are an empathetic, human-like conversational partner. Your goal is to establish "common ground" with the user regarding their emotional story. 
    
    CRITICAL RULES:
    1. Acknowledge their situation gracefully, but NEVER use the exact emotion labels provided in your hidden context (e.g., do not say "You are feeling anger/neutral").
    2. NEVER start your sentences with cliché therapy phrases like "It sounds like...", "I sense...", or "I hear you saying...". Speak naturally like a friend.
    3. Ask one gentle and simple clarification question to keep the narrative flowing.
    4. Keep your response strictly under 3 sentences."""
    
    summary_injection = f"\n\n[Running Summary (Semantic Memory)]:\n{current_summary}" if current_summary else ""
    instruction_injection = "\n\nInstructions: Formulate your next response by connecting their current text to the Running Summary (if available) to show you understand the bigger picture."
    
    chat_history[0]["content"] = base_system + summary_injection + instruction_injection

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
        memory_injection = ""
        if memory_data and memory_data.get("past_text"):
            past_text = memory_data["past_text"]
            past_emo = memory_data["past_top"]
            current_emo = final_emotion
            
            if memory_data.get("is_contradiction"):
                memory_injection = f"""
                [Hidden Memory Context]
                The user is talking about: "{transcription}" and currently feeling {current_emo}. 
                However, in the past, when they experienced a very similar event ("{past_text}"), they appeared to feel ({past_emo}). 
                Strategy: Gently note that they are reacting differently this time compared to the past, and ask a curious clarification question to explore why this time feels different to them.
                CRITICAL RULE: DO NOT use the literal words '{past_emo}' or '{current_emo}' in your response. Describe the shift using natural, empathetic human language.
                """
            else:
                memory_injection = f"""
                [Hidden Memory Context]
                The user is talking about: "{transcription}". 
                In the past, when they experienced a very similar event ("{past_text}"), they felt {past_emo}. 
                Today, they are feeling the exact same way ({current_emo}). 
                Strategy: Validate their feelings by explicitly acknowledging this pattern. Show them that it makes complete sense they feel this way again, and ask a gentle question to comfort them.
                CRITICAL RULE: DO NOT use the literal words '{past_emo}' or '{current_emo}' in your response. Paraphrase their emotional state using natural human language.
                """

        contextual_user_message = f"""
        [Hidden Context for Agent]
        User emotional profile: "{emotion_profile_text}"
        Detected emotional state: {final_emotion if final_emotion else "uncertain"}
        {memory_injection}

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
    tts_model.tts_to_file(text=sentence, file_path="output.wav")
    subprocess.run(["ffplay", "-nodisp", "-autoexit", "output.wav"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    print("📸 Initializing webcam (Please click 'OK' if Mac asks for permission)...")
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    stt_pipeline, text_emotion_pipeline, audeering_model, tts_model, device = model_initialization()
    db = PromptDatabase(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chroma_db'))
    
    chat_history = [{"role": "system", "content": "Initializing..."}]
    
    turn_counter = 1

    pending_clarification = None
    pending_event = None
    
    print("\n" + "="*60)
    print("✅ SYSTEM READY. Awaiting your turn.")
    print("="*60 + "\n")
    
    while True:

        print("\n" + "-"*60)
        user_cmd = input(f"🟢 TURN {turn_counter} | Press [ENTER] to start speaking (or type 'q' to quit): ")
        
       # --- SESSION SAVE ON QUIT ---
        if user_cmd.strip().lower() == 'q':
            print("\n👋 Wrapping up the conversation... Please wait a moment.")
            
            with summary_lock:
                final_summary = narrative_summary
                
            # --- GENERATE THE FINAL GOODBYE MESSAGE ---
            final_prompt = f"""
            [Hidden Context for Agent]
            The user has decided to end the conversation for today.
            
            Here is the running summary of their story and emotional journey today:
            {final_summary if final_summary else "(No summary available, it was a very short chat)."}
            
            Instructions:
            - Act as an empathetic therapist/friend saying goodbye.
            - Start by saying something like "Thank you so much for sharing all of this with me today."
            - Translate the summary above into a warm, natural narrative. You must explicitly mention the specific events they went through (the facts) and connect them to how they felt (the emotions).
            - Validate this specific emotional journey and reassure them.
            - End with a warm, encouraging sign-off.
            - Keep it compassionate and natural. Maximum 5 to 6 sentences.
            """
            
            chat_history.append({"role": "user", "content": final_prompt})
            payload = {"model": "llama3", "messages": chat_history, "stream": False, "think": False}
            
            try:
                response = requests.post(OLLAMA_URL, json=payload)
                farewell_msg = response.json().get("message", {}).get("content", "Thank you for chatting with me. Take care!")
                print("\n" + "="*60)
                print(f"💬 Agent: {farewell_msg}")
                print("="*60 + "\n")
            except Exception as e:
                print("\n💬 Agent: Thank you so much for chatting with me today. Take care of yourself!")

            # --- SAVE THE SUMMARY TO DISK ---
            if final_summary:
                filename = f"final_summary_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump({"semantic_summary": final_summary}, f, indent=4)
                print(f"💾 Saved Final Semantic Summary to {filename}")
                
            break # Exit the while loop
        
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
        if pending_clarification in ("conflict", "no_data"):
            print("🧠 Resolving previous emotional conflict/no_data from user reply...")

            final_emotion, final_distribution = resolve_conflict_with_user(
                transcription,
                text_emotion_pipeline
            )

            # 💡 FIX 1: Combine the two utterances into one "Document"
            # This ensures both the trigger and the explanation are searchable.
            full_narrative = f"Initial Event: {pending_event['initial_text']} | Clarification: {transcription}"

            emotions_record = {
                "final_emotion": final_emotion,
                "emotion_distribution": final_distribution
            }

            print(emotions_record)

            # 💡 FIX 2: Use the combined narrative as the primary Document
            memory_data = fetch_temporal_memory(db, transcription, final_distribution)
            db.add(full_narrative, emotions_record)

            emotion_profile_text = f"The user has clarified their feelings ({final_emotion}). Focus purely on the content of their explanation."

            print(f"💾 Stored resolved emotion → {final_emotion} ({final_distribution})")

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
                emotion_profile_text=emotion_profile_text,
                memory_data=memory_data
            )

            # 💡 Cleanly print the memory check for resolved turns
            if memory_data and memory_data.get("past_text"):
                print("\n🕰️ TEMPORAL MEMORY CHECK (Detour Resolved)")
                print(f"Past Event Matched: '{memory_data['past_text']}'")
                print(f"Vector Distance   : {memory_data['mae']:.3f} (MAE)")
                if memory_data['is_contradiction']:
                    print(f"Result            : ⚠️ CONTRADICTION (Threshold > {MEMORY_CONTRADICTION_THRESHOLD})")
                else:
                    print(f"Result            : ✅ ALIGNMENT (Threshold <= {MEMORY_CONTRADICTION_THRESHOLD})")

            print(f"🗣️ User Said: '{transcription}'")
            print(f"\n💬 Agent: {agent_reply}")
            text_to_speech(tts_model, agent_reply)
            # 💡 FIX: Add the resolution turn to the summary queue before continuing!
            turns_for_summary.append(f"User: {transcription}\nAgent: {agent_reply}")
            if len(turns_for_summary) >= 3:
                with summary_lock:
                    current_sum = narrative_summary
                summary_thread = threading.Thread(
                    target=update_running_summary, 
                    args=(list(turns_for_summary), current_sum)
                )
                summary_thread.daemon = True
                summary_thread.start()
                turns_for_summary.clear()

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

            pending_event = {
                "initial_text": transcription,
                "initial_state": "conflict"
            }

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

        # --- 5. EPISODIC MEMORY (CHROMA DB) WITH MAE MATH ---
        t0 = time.time()
        memory_data = {"past_text": None, "mae": 0.0, "is_contradiction": False, "past_top": None}
        
        # Only fetch memory if we actually have a confident present emotion!
        if final_emotion is not None and fused_dist:
            memory_data = fetch_temporal_memory(db, transcription, fused_dist)
            
        time_db = time.time() - t0

        # --- 6. THE LLM DIALOG MANAGER ---
        t0 = time.time()
        agent_reply = generate_agent_reply(
            transcription=transcription, 
            text_top=text_top, 
            final_emotion=final_emotion, 
            chat_history=chat_history, 
            decision=decision,
            modalities=modalities, 
            emotion_profile_text=emotion_profile_text, 
            memory_data=memory_data
        )
        time_llm = time.time() - t0

        print_final_output(transcription, top_3_text, arousal, valence, dominance,
                        ekman_probs_norm, avg_emotions_norm, valid_frames, agent_reply, text_confident, 
                        text_diff, audio_confident, audio_diff, decision, modalities, face_confident, face_diff, memory_data)
        save_debug_frames(video_frames, turn_counter)
        text_to_speech(tts_model, agent_reply)

        # --- HYBRID MEMORY: QUEUE FOR SUMMARY ---
        turns_for_summary.append(f"User: {transcription}\nAgent: {agent_reply}")
        if len(turns_for_summary) >= 3:
            with summary_lock:
                current_sum = narrative_summary
            summary_thread = threading.Thread(
                target=update_running_summary, 
                args=(list(turns_for_summary), current_sum)
            )
            summary_thread.daemon = True
            summary_thread.start()
            turns_for_summary.clear()

        # --- 7. STORE IN CHROMA ---
        if final_emotion is not None and pending_clarification is None:
            emotions_record = {
                "final_emotion": final_emotion,
                "emotion_distribution":  fused_dist
            }
            db.add(transcription, emotions_record)
            print(f"💾 Turn stored in Chroma (id: {transcription[:40]}...)")
            print(f"Stored FINAL emotion in Chroma → {final_emotion} ({fused_dist})")
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

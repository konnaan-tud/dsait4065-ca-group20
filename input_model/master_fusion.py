import cv2
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
import torch
import librosa
import requests
from transformers import pipeline
from deepface import DeepFace
from input_model.prosodic_modality.prosodic_abstraction import ProsodyEmotionPredictor
from input_model.prosodic_modality.test_audeering import Wav2Small 
from input_model.prosodic_modality.vad_mapping import VADEmotionMapper, load_vad_prototypes

# --- CONFIGURATION ---
AUDIO_FILE = "current_turn.wav"
SAMPLE_RATE = 16000
OLLAMA_URL = "http://localhost:11434/api/generate"
CONFIDENCE_THRESHOLD = 0.15 # to test

is_recording = False
video_frames = []
audio_data = []

print("📸 Initializing webcam (Please click 'OK' if Mac asks for permission)...")
cap = cv2.VideoCapture(0)
time.sleep(1)

# Checks if the emotion distribution is confident enough based on top1-top2 difference.
def is_confident(emotion_dict, threshold=CONFIDENCE_THRESHOLD):
    if not emotion_dict:
        return False, None, 0.0, 0.0

    sorted_emotions = sorted(emotion_dict.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_emotions) < 2:
        return False, sorted_emotions[0][0], sorted_emotions[0][1], 0.0

    top1_label, top1_score = sorted_emotions[0]
    top2_label, top2_score = sorted_emotions[1]

    diff = top1_score - top2_score
    confident = diff >= threshold

    return confident, top1_label, top1_score, diff

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
#stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small.en")
stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small.en")
print("  -> Loading RoBERTa Text Emotions...")
#text_emotion_pipeline = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
text_emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
print("  -> Loading Audeering Prosodic Emotions...")
# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
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
        prototypes=load_vad_prototypes(os.path.join(os.path.dirname(__file__), "prosodic_modality", "vad_mapping.csv")), # prototypes=load_vad_prototypes("vad_mapping.csv"),
        weights=(1.0,1.0,1.0),
        temperature=0.25
    )

    vt = threading.Thread(target=record_video)
    vt.daemon = True
    vt.start()

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
            print("\n🔴 Recording! Speak naturally.")
            print("🛑 Press [Ctrl+C] when finished...")
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
    text_emotions = {res["label"]: res["score"] for res in text_results}

    text_confident, text_top, text_score, text_diff = is_confident(text_emotions)

    top_3_text = [(res["label"], res["score"]) for res in text_results[:3]]
    time_roberta = time.time() - t0

    # 3. PROSODIC EMOTION (Audeering)
    t0 = time.time()

    signal = torch.from_numpy(librosa.load(AUDIO_FILE, sr=SAMPLE_RATE)[0])[None, :]
    with torch.no_grad():
        logits = audeering_model(signal.to(device))
    arousal, dominance, valence = logits[0, 0].item(), logits[0, 1].item(), logits[0, 2].item()
    ekman_probs = vad_mapper.predict_proba((valence, arousal, dominance))

    audio_confident, audio_top, audio_score, audio_diff = is_confident(ekman_probs)
    time_audeering = time.time() - t0

    # 4. FACIAL EMOTION (DeepFace)
    t0 = time.time()
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

        face_confident, top_face_emo, face_score, face_diff = is_confident(avg_emotions)
    else:
        avg_emotions=sum_emotions
        face_confident=False
        top_face_emo="No face detected"
        face_diff=0
    time_deepface = time.time() - t0

    cap.release()

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

    # Defines agreement and detects conflict
    def analyze_agreement(modalities):

        if len(modalities) == 0:
            return "no_data", None

        tops = [m["top"] for m in modalities.values()]
        counts = {e: tops.count(e) for e in set(tops)}

        max_count = max(counts.values())
        dominant = max(counts, key=counts.get)

        if max_count == len(modalities):
            return "full_agreement", dominant
        elif max_count == 2:
            return "partial_agreement", dominant
        else:
            return "conflict", None

    # Applies weighted linear fusion
    def fuse_modalities(modalities):
        emotions = list(next(iter(modalities.values()))["probs"].keys())

        c = {m: modalities[m]["confidence"] for m in modalities}
        total_c = sum(c.values())

        weights = {m: c[m] / total_c for m in modalities}

        fused = {e: 0.0 for e in emotions}

        for m in modalities:
            probs = modalities[m]["probs"]
            for e in emotions:
                fused[e] += weights[m] * probs[e]

        final_emotion = max(fused, key=fused.get)

        return final_emotion, fused, weights


    # Define prompts for each case

    decision, agreed_emotion = analyze_agreement(modalities)

    # CASE 0: No confident modality
    if decision == "no_data":
        final_emotion = None
        emotion_profile_text = "No confident emotional signal detected." # TODO: Change text


    # CASE 1: ONLY ONE CONFIDENT MODALITY  🔥 (NEW RULE)
    elif len(modalities) == 1:
        m_name = list(modalities.keys())[0]
        m = modalities[m_name]

        final_emotion = m["top"]

        emotion_profile_text = (
            f"- Single confident modality used: {m_name}\n"
            f"- Detected emotion: {final_emotion}"
        ) # TODO: Change text

    # CASE 2: CONFLICT (ALL DIFFERENT)
    elif decision == "conflict":
        final_emotion = None
        emotion_profile_text = "Conflicting emotional signals across modalities." # TODO: Change text

    # CASE 3: AGREEMENT/PARTIAL AGREEMENT → FUSION
    else:
        final_emotion, fused_dist, weights = fuse_modalities(modalities)

        emotion_profile = [f"- Final emotion (fused): {final_emotion}"]

        for m in modalities:
            emotion_profile.append(
                f"- {m.capitalize()} supports: {modalities[m]['top']} "
                f"(weight={weights[m]:.2f})"
            )

        emotion_profile_text = "\n".join(emotion_profile)

    # --- 5. THE LLM DIALOG MANAGER ---
    print("\n🧠 Sending profile to Llama...")

    if decision == "conflict":

        system_prompt = f"""
    You are an empathetic conversational agent.

    The user said:
    "{transcription}"

    The emotional signals are conflicting across modalities.

    Ask the user for clarification about how they feel.
    """

    elif decision == "no_data":

        system_prompt = f"""
    You are an empathetic conversational agent.

    The user said:
    "{transcription}"

    No strong emotional signal was detected.

    Respond naturally and gently.
    """

    else:

        system_prompt = f"""
    You are an empathetic conversational agent.

    The user said:
    "{transcription}"

    Here is the emotional analysis:
    {emotion_profile_text}

    Respond empathetically in 2-3 sentences.
    """
        
    payload = { "model":"llama3.2:1b", "prompt":system_prompt, "stream":False } 
    t0 = time.time() 
    try: 
        response = requests.post(OLLAMA_URL, json=payload) 
        agent_reply = response.json().get( "response", "Error generating response." ) 
    except Exception as e: 
        agent_reply = f"Could not connect: {e}" 
        time_llm = time.time() - t0
        
    # --- FINAL OUTPUT ---
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
    
    print("="*60)

    os.makedirs("debug_frames", exist_ok=True)
    for i, frame in enumerate(video_frames):
        cv2.imwrite(f"debug_frames/face_second_{i+1}.jpg", frame)
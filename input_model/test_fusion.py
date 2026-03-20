import cv2
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
import torch
import librosa
import requests
from collections import Counter
from transformers import pipeline
from deepface import DeepFace
from scipy.spatial.distance import jensenshannon
from test_audeering import Wav2Small

# ============================================================
# CONFIGURATION
# ============================================================
AUDIO_FILE   = "current_turn.wav"
SAMPLE_RATE  = 16000
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"

JS_TAU = 0.40   # conflict threshold used only when no majority exists (JS² range: 0–1)

EKMAN = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

# ============================================================
# GLOBAL STATE
# ============================================================
is_recording = False
video_frames = []
audio_data   = []

# ============================================================
# CAMERA INIT
# ============================================================
print("📸 Initializing webcam...")
cap = cv2.VideoCapture(0)
time.sleep(1)
if not cap.isOpened():
    print("❌ Camera failed to open. Check permissions.")
else:
    ret, test_frame = cap.read()
    print("✅ Camera OK." if ret else "⚠️  Camera opened but first read failed.")

# ============================================================
# THREAD: VIDEO RECORDER
# ============================================================
def record_video():
    global is_recording, video_frames, cap
    last_capture_time = 0
    frame_count = 0
    while is_recording:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        now = time.time()
        if now - last_capture_time >= 1.0:        # 1 frame per second
            video_frames.append(frame.copy())
            frame_count += 1
            last_capture_time = now
            print(f"  📸 Frame {frame_count} stored", end="\r")
        time.sleep(0.03)                           # ~30 fps read rate

# ============================================================
# THREAD: AUDIO RECORDER
# ============================================================
def audio_callback(indata, frames, time_info, status):
    if is_recording:
        audio_data.append(indata.copy())

# ============================================================
# GOEMOTION → EKMAN MAPPING
# ============================================================
GOEMOTION_TO_EKMAN = {
    "anger": "anger", "annoyance": "anger",
    "disgust": "disgust",
    "fear": "fear", "nervousness": "fear",
    "joy": "joy", "amusement": "joy", "approval": "joy",
    "gratitude": "joy", "love": "joy", "optimism": "joy",
    "sadness": "sadness", "disappointment": "sadness",
    "grief": "sadness", "remorse": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}

def text_to_ekman(text_results):
    probs = dict.fromkeys(EKMAN, 0.0)
    for r in text_results:
        ekman_label = GOEMOTION_TO_EKMAN.get(r["label"])
        if ekman_label:
            probs[ekman_label] += r["score"]
    total = sum(probs.values()) + 1e-8
    return {k: v / total for k, v in probs.items()}

# ============================================================
# VAD → EKMAN MAPPING
# ============================================================
def vad_to_ekman(valence, arousal):
    """Maps Audeering VAD (valence, arousal ∈ [-1,1]) to Ekman 7 emotions."""
    v_pos = max(0.0, valence)
    v_neg = max(0.0, -valence)
    a_pos = max(0.0, arousal)
    a_neg = max(0.0, -arousal)
    v_abs = abs(valence)

    scores = {
        "joy":      v_pos * a_pos,
        "anger":    v_neg * a_pos,
        "fear":     v_neg * a_pos,   # fear and anger share quadrant
        "sadness":  v_neg * a_neg,
        "surprise": (1 - v_abs) * a_pos,
        "neutral":  (1 - v_abs) * a_neg,
        "disgust":  v_neg * 0.5,
    }
    total = sum(scores.values()) + 1e-8
    return {k: v / total for k, v in scores.items()}

# ============================================================
# DEEPFACE → EKMAN MAPPING
# ============================================================
DEEPFACE_TO_EKMAN = {
    "angry":    "anger",
    "disgust":  "disgust",
    "fear":     "fear",
    "happy":    "joy",
    "sad":      "sadness",
    "surprise": "surprise",
    "neutral":  "neutral",
}

def analyze_face_frames(frames):
    """
    Runs DeepFace on every captured frame, averages the results.
    Returns (ekman_probs dict, valid_frame_count).
    """
    ekman_sum = dict.fromkeys(EKMAN, 0.0)
    valid = 0

    for i, frame in enumerate(frames):
        try:
            result = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )[0]
            for df_label, score in result["emotion"].items():
                ek = DEEPFACE_TO_EKMAN.get(df_label)
                if ek:
                    ekman_sum[ek] += score / 100.0   # DeepFace returns percentages
            valid += 1
            print(f"  Face frame {i+1}: dominant = {result['dominant_emotion']}")
        except Exception as e:
            print(f"  Face frame {i+1}: skipped — {e}")

    if valid > 0:
        avg = {k: v / valid for k, v in ekman_sum.items()}
        total = sum(avg.values()) + 1e-8
        return {k: v / total for k, v in avg.items()}, valid

    print("⚠️  No face detected — using neutral fallback.")
    fallback = dict.fromkeys(EKMAN, 0.0)
    fallback["neutral"] = 1.0
    return fallback, 0

# ============================================================
# JS² HELPER
# ============================================================
def _js2(p_dict, q_dict):
    """JS divergence squared (bounded 0–1). scipy jensenshannon returns sqrt."""
    p = np.array([p_dict[e] for e in EKMAN], dtype=float)
    q = np.array([q_dict[e] for e in EKMAN], dtype=float)
    return float(jensenshannon(p, q) ** 2)

# ============================================================
# CONTRADICTION DETECTION (MAJORITY VOTE + JS² FALLBACK)
# ============================================================
def detect_contradiction(text_p, prosody_p, face_p, face_is_real=True):
    """
    Decision logic (in priority order):

      1. 3/3 dominant emotions agree  → AGREE, return majority emotion
      2. 2/3 dominant emotions agree  → AGREE, return majority emotion
      3. 0/3 agree (all different)    → compute JS² across all pairs;
                                        if worst-case JS² > JS_TAU → CONFLICT
                                        else → AGREE, fuse all three

    Returns:
      (worst_js_score, conflict_bool, majority_emotion_or_None)

    JS² is only ever computed in case 3.
    Face is excluded from voting when face_is_real=False.
    """
    dom_text    = max(text_p,    key=text_p.get)
    dom_prosody = max(prosody_p, key=prosody_p.get)
    dom_face    = max(face_p,    key=face_p.get) if face_is_real else None

    active = {"text": dom_text, "prosody": dom_prosody}
    if face_is_real:
        active["face"] = dom_face

    print(f"  Dominant — text:{dom_text}  prosody:{dom_prosody}" +
          (f"  face:{dom_face}" if face_is_real else "  face:N/A (fallback)"))

    votes = Counter(active.values())
    majority_emotion, majority_count = votes.most_common(1)[0]
    total_active = len(active)

    # ── Case 1: full agreement ─────────────────────────────
    if majority_count == total_active:
        print(f"  ✅ Full agreement ({total_active}/{total_active}): {majority_emotion}")
        return 0.0, False, majority_emotion

    # ── Case 2: majority (2/3) ─────────────────────────────
    if majority_count >= 2:
        winners = [m for m, e in active.items() if e == majority_emotion]
        dissenter = [m for m, e in active.items() if e != majority_emotion]
        print(f"  ✅ Majority agreement [{', '.join(winners)}]: {majority_emotion}"
              f"  (dissenter: {dissenter[0]}={active[dissenter[0]]})")
        return 0.0, False, majority_emotion

    # ── Case 3: no majority — fall back to JS² ────────────
    print("  ⚠️  No majority — computing JS² divergence...")
    pairs = [("text↔prosody", text_p, prosody_p)]
    if face_is_real:
        pairs.append(("text↔face",    text_p,    face_p))
        pairs.append(("prosody↔face", prosody_p, face_p))

    scores = {label: _js2(p, q) for label, p, q in pairs}
    for label, score in scores.items():
        print(f"  JS² {label}: {score:.3f}")

    worst_label = max(scores, key=scores.get)
    worst_score = scores[worst_label]
    print(f"  Worst pair: {worst_label} = {worst_score:.3f} (τ={JS_TAU})")
    return worst_score, worst_score > JS_TAU, None

# ============================================================
# LATE FUSION (CONFIDENCE-WEIGHTED)
# ============================================================
def fuse(text_p, prosody_p, face_p, face_is_real=True):
    """
    Weights each modality by its peak confidence, then does a weighted sum.
    Excludes face when it is a neutral fallback.
    """
    modalities = {"text": text_p, "prosody": prosody_p}
    if face_is_real:
        modalities["face"] = face_p

    confidences  = {m: max(p.values()) for m, p in modalities.items()}
    total_conf   = sum(confidences.values()) + 1e-8
    weights      = {m: c / total_conf for m, c in confidences.items()}

    fused = dict.fromkeys(EKMAN, 0.0)
    for m, probs in modalities.items():
        for e in EKMAN:
            fused[e] += weights[m] * probs[e]

    final = max(fused, key=fused.get)
    return final, fused, weights

# ============================================================
# LLM CALL
# ============================================================
def call_llm(prompt):
    try:
        payload  = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        return response.json().get("response", "Error generating response.")
    except Exception as e:
        return f"Could not connect to Ollama: {e}"

# ============================================================
# MODEL INITIALIZATION
# ============================================================
print("\n🧠 Loading models... (10–15 seconds)\n")

print("  -> Whisper (speech-to-text)...")
stt = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small.en",
    device=0,
)

print("  -> RoBERTa (text emotions)...")
text_clf = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,
    device=0,
)

print("  -> Audeering (prosody VAD)...")
device   = "cuda" if torch.cuda.is_available() else "cpu"
audeering = Wav2Small.from_pretrained("audeering/wav2small").to(device).eval()

print("  -> Pre-warming DeepFace...")
try:
    dummy = np.zeros((48, 48, 3), dtype=np.uint8)
    DeepFace.analyze(img_path=dummy, actions=["emotion"], enforce_detection=False, silent=True)
    print("  -> DeepFace ready.")
except Exception as e:
    print(f"  -> DeepFace warmup note: {e}")

print("\n" + "=" * 60)
print("✅ SYSTEM READY.")
print("=" * 60 + "\n")

# ============================================================
# MAIN LOOP
# ============================================================
if __name__ == "__main__":
    input("🎤 Press [ENTER] to start your turn...")

    is_recording = True
    video_frames.clear()
    audio_data.clear()

    vt = threading.Thread(target=record_video, daemon=True)
    vt.start()

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
            print("\n🔴 Recording — speak naturally.")
            print("🛑 Press [Ctrl+C] to stop...\n")
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\n✅ Recording stopped.")
        is_recording = False

    vt.join(timeout=2.0)

    full_audio = np.concatenate(audio_data, axis=0)
    sf.write(AUDIO_FILE, full_audio, SAMPLE_RATE)
    print(f"\n💾 Audio saved ({len(video_frames)} video frames captured)")

    # ----------------------------------------------------------
    # 1. Transcription
    # ----------------------------------------------------------
    print("\n🔤 Transcribing...")
    t0 = time.time()
    transcription = stt(AUDIO_FILE)["text"].strip()
    t_whisper = time.time() - t0
    print(f"   '{transcription}'  ({t_whisper:.2f}s)")

    # ----------------------------------------------------------
    # 2. Text emotion
    # ----------------------------------------------------------
    print("\n📖 Text emotion (RoBERTa)...")
    t0 = time.time()
    text_results = text_clf(transcription)[0]
    text_probs   = text_to_ekman(text_results)
    t_roberta    = time.time() - t0
    for e in EKMAN:
        bar = "█" * int(text_probs[e] * 20)
        print(f"   {e:<10} {text_probs[e]:.2f}  {bar}")
    print(f"   ({t_roberta:.2f}s)")

    # ----------------------------------------------------------
    # 3. Prosody emotion
    # ----------------------------------------------------------
    print("\n🎵 Prosody emotion (Audeering)...")
    t0 = time.time()
    signal = torch.from_numpy(librosa.load(AUDIO_FILE, sr=SAMPLE_RATE)[0])[None, :]
    with torch.no_grad():
        logits = audeering(signal.to(device))
    arousal, dominance, valence = logits[0, 0].item(), logits[0, 1].item(), logits[0, 2].item()
    prosody_probs = vad_to_ekman(valence, arousal)
    t_audeering   = time.time() - t0
    print(f"   Valence={valence:.2f}  Arousal={arousal:.2f}  Dominance={dominance:.2f}")
    for e in EKMAN:
        bar = "█" * int(prosody_probs[e] * 20)
        print(f"   {e:<10} {prosody_probs[e]:.2f}  {bar}")
    print(f"   ({t_audeering:.2f}s)")

    # ----------------------------------------------------------
    # 4. Face emotion
    # ----------------------------------------------------------
    print(f"\n🎭 Face emotion (DeepFace — {len(video_frames)} frames)...")
    t0 = time.time()
    face_probs, valid_frames = analyze_face_frames(video_frames)
    t_deepface  = time.time() - t0
    face_is_real = valid_frames > 0
    for e in EKMAN:
        bar = "█" * int(face_probs[e] * 20)
        print(f"   {e:<10} {face_probs[e]:.2f}  {bar}")
    print(f"   ({t_deepface:.2f}s, {valid_frames} valid frames)")

    cap.release()

    # ----------------------------------------------------------
    # 5. Contradiction detection (majority vote + JS² fallback)
    # ----------------------------------------------------------
    print("\n🔎 Checking for contradiction...")
    disagreement, conflict, majority_emotion = detect_contradiction(
        text_probs, prosody_probs, face_probs,
        face_is_real=face_is_real,
    )
    print(f"   → {'⚠️  CONFLICT' if conflict else '✅ AGREEMENT'}"
          + (f" on '{majority_emotion}'" if majority_emotion else
             f"  JS²={disagreement:.3f}"))

    # ----------------------------------------------------------
    # 6. Build LLM prompt
    # ----------------------------------------------------------
    t0 = time.time()

    if conflict:
        # All three modalities disagree AND JS² exceeds threshold
        prompt = f"""The user said: "{transcription}"

Their emotional signals conflict:
- Text suggests: {max(text_probs,    key=text_probs.get)}
- Voice suggests: {max(prosody_probs, key=prosody_probs.get)}
- Face suggests: {max(face_probs,    key=face_probs.get)}

Ask ONE short, natural question to clarify how they are actually feeling.
Do not mention models, signals, or technology.
Reply with only the question — no preamble."""

    else:
        if majority_emotion:
            # Majority vote path — agreed emotion is already known
            final_emotion = majority_emotion
            print(f"\n⚖️  Majority emotion used: {final_emotion}")
        else:
            # No majority but JS² below threshold — safe to fuse all three
            final_emotion, fused_probs, weights = fuse(
                text_probs, prosody_probs, face_probs,
                face_is_real=face_is_real,
            )
            weight_str = "  ".join(f"{m}={w:.2f}" for m, w in weights.items())
            print(f"\n⚖️  No majority, JS² low → fused  [{weight_str}]")
            print(f"   Final emotion: {final_emotion}")

        prompt = f"""The user said: "{transcription}"

Their detected emotion is: {final_emotion}

Respond empathetically in 2 sentences."""

    # ----------------------------------------------------------
    # 7. LLM response
    # ----------------------------------------------------------
    print("\n🧠 Asking LLM...")
    agent_reply = call_llm(prompt)
    t_llm = time.time() - t0

    # ----------------------------------------------------------
    # 8. Final output
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("🤖 AGENT RESPONSE")
    print("=" * 60)
    print(f"🗣️  User:  '{transcription}'")
    print(f"💬 Agent: {agent_reply}")
    print("=" * 60)

    print("\n⏱️  LATENCY")
    print(f"   Whisper   : {t_whisper:.2f}s")
    print(f"   RoBERTa   : {t_roberta:.2f}s")
    print(f"   Audeering : {t_audeering:.2f}s")
    print(f"   DeepFace  : {t_deepface:.2f}s  ({valid_frames} frames)")
    print(f"   LLM       : {t_llm:.2f}s")
    total = t_whisper + t_roberta + t_audeering + t_deepface + t_llm
    print(f"   ─────────────────")
    print(f"   TOTAL     : {total:.2f}s")

    # ----------------------------------------------------------
    # 9. Save debug frames
    # ----------------------------------------------------------
    import os
    os.makedirs("debug_frames", exist_ok=True)
    for i, frame in enumerate(video_frames):
        cv2.imwrite(f"debug_frames/frame_{i+1:03d}.jpg", frame)
    print(f"\n🖼️  {len(video_frames)} debug frames saved to debug_frames/")
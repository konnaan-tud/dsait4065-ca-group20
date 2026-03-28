import os
import sys
import time
import json
import html
import queue
import signal
import threading
import subprocess
from pathlib import Path
from collections import deque

import streamlit as st


# ============================================================
# CONFIG
# ============================================================
TARGET_SCRIPT = "master_fusion_basic.py"   # <-- change if needed
ROOT_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = ROOT_DIR / TARGET_SCRIPT
MAX_LOG_LINES = 3000
MAX_CHAT_MESSAGES = 200
AUTO_REFRESH_SECONDS = 0.7


# ============================================================
# STREAMLIT SETUP
# ============================================================
st.set_page_config(
    page_title="Multimodal Agent Dashboard",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================
def init_state():
    defaults = {
        "process": None,
        "reader_thread": None,
        "stderr_thread": None,
        "stdout_queue": queue.Queue(),
        "stderr_queue": queue.Queue(),
        "process_running": False,
        "logs": deque(maxlen=MAX_LOG_LINES),
        "events": [],
        "chat_messages": [],
        "current_turn_id": None,
        "latest_status": "Ready",
        "latest_transcription": None,
        "latest_agent_reply": None,
        "show_debug": False,
        "interaction_state": "idle",  # idle|starting|awaiting_start|recording|processing|stopping
        "pending_farewell": False,
        "current_turn": {
            "turn_id": None,
            "transcription": None,
            "agent_reply": None,
            "text_top3": [],
            "audio_probs": {},
            "video_probs": {},
            "confidence": {},
            "decision": None,
            "modalities": {},
            "latency": {},
            "valid_frames": None,
            "running_summary": None,
        },
        "completed_turns": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ============================================================
# HELPERS
# ============================================================
def log(line: str, source: str = "stdout"):
    if line is None:
        return
    line = line.rstrip("\n")
    if not line:
        return
    prefix = "[stderr] " if source == "stderr" else ""
    st.session_state.logs.append(prefix + line)


def reader(pipe, out_queue):
    try:
        for line in iter(pipe.readline, ""):
            out_queue.put(line)
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def safe_json_loads(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        return None


def clear_queue(q: queue.Queue):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def add_chat_message(role: str, text: str, turn_id=None):
    if not text:
        return

    if st.session_state.chat_messages:
        last = st.session_state.chat_messages[-1]
        if last.get("role") == role and last.get("text") == text and last.get("turn_id") == turn_id:
            return

    st.session_state.chat_messages.append(
        {
            "role": role,
            "text": text,
            "turn_id": turn_id,
        }
    )
    if len(st.session_state.chat_messages) > MAX_CHAT_MESSAGES:
        st.session_state.chat_messages = st.session_state.chat_messages[-MAX_CHAT_MESSAGES:]


def reset_chat():
    prev_status = st.session_state.latest_status
    prev_interaction = st.session_state.interaction_state
    st.session_state.chat_messages = []
    st.session_state.latest_transcription = None
    st.session_state.latest_agent_reply = None
    st.session_state.current_turn_id = None
    st.session_state.pending_farewell = False
    clear_queue(st.session_state.stdout_queue)
    clear_queue(st.session_state.stderr_queue)
    if st.session_state.process_running:
        st.session_state.latest_status = prev_status
        st.session_state.interaction_state = prev_interaction
    else:
        st.session_state.latest_status = "Ready"
        st.session_state.interaction_state = "idle"


def is_waiting_for_agent_reply() -> bool:
    if (
        st.session_state.process_running
        and st.session_state.interaction_state == "stopping"
        and st.session_state.pending_farewell
    ):
        return True

    turn_id = st.session_state.current_turn_id
    if turn_id is None:
        return False

    has_user = any(
        m.get("role") == "user" and m.get("turn_id") == turn_id
        for m in st.session_state.chat_messages
    )
    has_assistant = any(
        m.get("role") == "assistant" and m.get("turn_id") == turn_id
        for m in st.session_state.chat_messages
    )

    return (
        st.session_state.process_running
        and st.session_state.interaction_state == "processing"
        and has_user
        and not has_assistant
    )


def push_completed_turn_if_ready():
    cur = st.session_state.current_turn
    if cur.get("transcription") or cur.get("agent_reply") or cur.get("latency"):
        snapshot = json.loads(json.dumps(cur))
        # avoid duplicating the same object repeatedly
        if not st.session_state.completed_turns or st.session_state.completed_turns[-1] != snapshot:
            st.session_state.completed_turns.append(snapshot)
            if len(st.session_state.completed_turns) > 20:
                st.session_state.completed_turns = st.session_state.completed_turns[-20:]


def handle_event(event: dict):
    etype = event.get("type")
    st.session_state.events.append(event)
    if len(st.session_state.events) > 1000:
        st.session_state.events = st.session_state.events[-1000:]

    if etype == "awaiting_start":
        st.session_state.current_turn_id = event.get("turn_id")
        st.session_state.latest_status = "Ready to start speaking"
        st.session_state.interaction_state = "awaiting_start"
        st.session_state.pending_farewell = False
        return

    cur = st.session_state.current_turn

    if etype == "turn_start":
        if cur.get("transcription") or cur.get("agent_reply"):
            push_completed_turn_if_ready()
        st.session_state.current_turn_id = event.get("turn_id")
        st.session_state.latest_status = "Listening"
        st.session_state.interaction_state = "recording"
        st.session_state.pending_farewell = False
        st.session_state.current_turn = {
            "turn_id": event.get("turn_id"),
            "transcription": None,
            "agent_reply": None,
            "text_top3": [],
            "audio_probs": {},
            "video_probs": {},
            "confidence": {},
            "decision": None,
            "modalities": {},
            "latency": {},
            "valid_frames": None,
            "running_summary": None,
        }
        return

    cur = st.session_state.current_turn

    if etype == "transcription":
        text = event.get("text")
        cur["transcription"] = text
        st.session_state.latest_transcription = text
        st.session_state.latest_status = "Generating response"
        st.session_state.interaction_state = "processing"
        add_chat_message("user", text, st.session_state.current_turn_id)
    elif etype == "agent_reply":
        text = event.get("text")
        cur["agent_reply"] = text
        st.session_state.latest_agent_reply = text
        st.session_state.latest_status = "Agent replied"
        st.session_state.pending_farewell = False
        add_chat_message("assistant", text, st.session_state.current_turn_id)
    elif etype == "text_top3":
        cur["text_top3"] = event.get("items", [])
    elif etype == "audio_probs":
        cur["audio_probs"] = event.get("items", {})
    elif etype == "video_probs":
        cur["video_probs"] = event.get("items", {})
        cur["valid_frames"] = event.get("valid_frames")
    elif etype == "confidence":
        cur["confidence"] = event.get("items", {})
    elif etype == "decision":
        cur["decision"] = event.get("decision")
        cur["modalities"] = event.get("modalities", {})
    elif etype == "latency":
        cur["latency"] = event.get("items", {})
        st.session_state.latest_status = "Turn complete"
        push_completed_turn_if_ready()
    elif etype == "running_summary":
        cur["running_summary"] = event.get("text")


# Lines beginning with this prefix are treated as structured events.
EVENT_PREFIX = "UI_EVENT::"


def pump_queues():
    changed = False

    while True:
        try:
            raw = st.session_state.stdout_queue.get_nowait()
        except queue.Empty:
            break

        if raw.startswith(EVENT_PREFIX):
            payload = raw[len(EVENT_PREFIX):].strip()
            event = safe_json_loads(payload)
            if event is not None:
                handle_event(event)
                changed = True
            else:
                log(raw)
                changed = True
        else:
            log(raw)
            changed = True

    while True:
        try:
            raw = st.session_state.stderr_queue.get_nowait()
        except queue.Empty:
            break
        log(raw, "stderr")
        changed = True

    proc = st.session_state.process
    if proc is not None and proc.poll() is not None and st.session_state.process_running:
        st.session_state.process_running = False
        st.session_state.interaction_state = "idle"
        st.session_state.pending_farewell = False
        st.session_state.latest_status = f"Process exited ({proc.returncode})"
        log(f"[system] Process exited with code {proc.returncode}")
        push_completed_turn_if_ready()
        changed = True

    return changed


def start_process():
    if not SCRIPT_PATH.exists():
        log(f"[system] Script not found: {SCRIPT_PATH}")
        return

    if st.session_state.process_running:
        log("[system] Process is already running.")
        return

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=str(ROOT_DIR),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

    st.session_state.process = process
    st.session_state.process_running = True
    st.session_state.latest_status = "Starting agent"
    st.session_state.interaction_state = "starting"

    st.session_state.reader_thread = threading.Thread(
        target=reader,
        args=(process.stdout, st.session_state.stdout_queue),
        daemon=True,
    )
    st.session_state.stderr_thread = threading.Thread(
        target=reader,
        args=(process.stderr, st.session_state.stderr_queue),
        daemon=True,
    )
    st.session_state.reader_thread.start()
    st.session_state.stderr_thread.start()

    log(f"[system] Started process: {SCRIPT_PATH.name}")


def stop_process():
    proc = st.session_state.process
    if proc is None or not st.session_state.process_running:
        log("[system] No running process to stop.")
        return

    try:
        if os.name == "nt":
            proc.terminate()
        else:
            proc.send_signal(signal.SIGTERM)
        st.session_state.latest_status = "Stopping agent"
        st.session_state.interaction_state = "stopping"
        log("[system] Sent terminate signal.")
    except Exception as exc:
        log(f"[system] Failed to stop process cleanly: {exc}")


def send_stdin(text: str):
    proc = st.session_state.process
    if proc is None or not st.session_state.process_running:
        log("[system] Cannot send input because the process is not running.")
        return
    try:
        proc.stdin.write(text)
        proc.stdin.flush()
        visible = text.replace("\n", "\\n")
        log(f"[ui → process] {visible}")
    except Exception as exc:
        log(f"[system] Failed to send stdin: {exc}")


def start_speaking():
    st.session_state.latest_status = "Listening"
    st.session_state.interaction_state = "recording"
    send_stdin("\n")


def stop_speaking():
    st.session_state.latest_status = "Processing"
    st.session_state.interaction_state = "processing"
    send_stdin("\n")


def quit_gracefully():
    st.session_state.latest_status = "Quitting"
    st.session_state.interaction_state = "stopping"
    st.session_state.pending_farewell = True
    send_stdin("q\n")


# ============================================================
# UI
# ============================================================
pump_queues()
ui_action_taken = False

st.markdown(
    """
    <style>
        .study-shell {
            max-width: 760px;
            margin: 0 auto;
            padding-top: 0.5rem;
        }
        .study-title {
            text-align: center;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .study-subtitle {
            text-align: center;
            color: #6b7280;
            margin-bottom: 1rem;
        }
        .status-pill {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: #f3f4f6;
            border: 1px solid #e5e7eb;
            font-size: 0.9rem;
            margin-bottom: 0.75rem;
            color: black;
        }
        .control-hint {
            text-align: center;
            color: #6b7280;
            font-size: 0.95rem;
            margin-top: 0.5rem;
        }
        .typing {
            display: inline-flex;
            gap: 0.3rem;
            align-items: center;
            min-height: 1.4rem;
        }
        .typing span {
            width: 0.45rem;
            height: 0.45rem;
            background: #9ca3af;
            border-radius: 50%;
            animation: blink 1s infinite ease-in-out;
        }
        .typing span:nth-child(2) {
            animation-delay: 0.15s;
        }
        .typing span:nth-child(3) {
            animation-delay: 0.3s;
        }
        @keyframes blink {
            0%, 80%, 100% { transform: scale(0.7); opacity: 0.45; }
            40% { transform: scale(1); opacity: 1; }
        }
        .raw-console-scroll {
            max-height: 300px;
            overflow-y: auto;
            overflow-x: auto;
            background: #111827;
            color: #e5e7eb;
            border-radius: 0.5rem;
            padding: 0.75rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 0.82rem;
            line-height: 1.35;
            white-space: pre;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="study-shell">', unsafe_allow_html=True)
st.markdown('<div class="study-title">Conversation Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="study-subtitle">Speak naturally and continue the conversation using the buttons below.</div>', unsafe_allow_html=True)

status_text = st.session_state.latest_status
st.markdown(f'<div style="text-align:center;"><span class="status-pill">Status: {status_text}</span></div>', unsafe_allow_html=True)

chat_box = st.container(height=460, border=True)
with chat_box:
    if st.session_state.chat_messages:
        for message in st.session_state.chat_messages:
            avatar = "🧑" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.write(message["text"])

        if is_waiting_for_agent_reply():
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(
                    '<div class="typing"><span></span><span></span><span></span></div>',
                    unsafe_allow_html=True,
                )
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write("Hello — press **Start speaking** when you are ready.")

st.write("")

col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    can_start = st.session_state.process_running and st.session_state.interaction_state == "awaiting_start"
    if st.button(
        "Start speaking",
        width="stretch",
        disabled=not can_start,
        type="primary",
    ):
        ui_action_taken = True
        start_speaking()

with col2:
    can_stop = st.session_state.process_running and st.session_state.interaction_state == "recording"
    if st.button(
        "Stop speaking",
        width="stretch",
        disabled=not can_stop,
    ):
        ui_action_taken = True
        stop_speaking()

with col3:
    can_end = st.session_state.process_running and st.session_state.interaction_state == "awaiting_start"
    if st.button(
        "End conversation",
        width="stretch",
        disabled=not can_end,
    ):
        ui_action_taken = True
        quit_gracefully()

st.markdown('<div class="control-hint">Press start, speak, then press stop when you are done.</div>', unsafe_allow_html=True)

st.write("")

with st.expander("Session controls", expanded=False):
    a, b = st.columns(2, gap="small")
    with a:
        if st.button("Launch agent", width="stretch"):
            ui_action_taken = True
            start_process()
    with b:
        if st.button("Stop process", width="stretch", disabled=not st.session_state.process_running):
            ui_action_taken = True
            stop_process()

d, = st.columns(1, gap="small")
with d:
    if st.button("Reset chat view", width="stretch"):
        ui_action_taken = True
        reset_chat()

st.session_state.show_debug = True

if st.session_state.show_debug:
    st.write("")
    st.subheader("Debug panel")

    p1, p2, p3 = st.columns(3)
    p1.metric("Process running", "Yes" if st.session_state.process_running else "No")
    p2.metric("Messages", len(st.session_state.chat_messages))
    p3.metric("Logs", len(st.session_state.logs))

    current = st.session_state.current_turn
    top_a, top_b = st.columns([2, 1], gap="large")

    with top_a:
        st.subheader("Current turn")
        with st.container(border=True):
            st.markdown(f"**Turn:** {current.get('turn_id') if current.get('turn_id') is not None else '-'}")
            st.markdown("**Transcription**")
            st.write(current.get("transcription") or "Waiting for transcription...")
            st.markdown("**Agent reply**")
            st.write(current.get("agent_reply") or "Waiting for reply...")
            st.markdown("**Decision**")
            st.write(current.get("decision") or "-")
            if current.get("running_summary"):
                with st.expander("Running summary", expanded=False):
                    st.write(current["running_summary"])

    with top_b:
        st.subheader("Confidence")
        with st.container(border=True):
            conf = current.get("confidence", {})
            if conf:
                for name, info in conf.items():
                    st.write(f"**{name.capitalize()}** — confident: {info.get('confident')} | diff: {info.get('diff')}")
            else:
                st.write("No confidence data yet.")

    b1, b2, b3 = st.columns(3, gap="large")

    with b1:
        st.markdown("### Text emotion")
        with st.container(border=True):
            items = current.get("text_top3", [])
            if items:
                for item in items:
                    st.write(f"**{item['label']}**: {item['score']:.3f}")
            else:
                st.write("No text emotion output yet.")

    with b2:
        st.markdown("### Audio emotion")
        with st.container(border=True):
            items = current.get("audio_probs", {})
            if items:
                sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)
                for emo, score in sorted_items[:5]:
                    st.write(f"**{emo}**: {score:.3f}")
            else:
                st.write("No audio emotion output yet.")

    with b3:
        st.markdown("### Video emotion")
        with st.container(border=True):
            items = current.get("video_probs", {})
            valid_frames = current.get("valid_frames")
            if valid_frames is not None:
                st.write(f"Frames processed: **{valid_frames}**")
            if items:
                sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)
                for emo, score in sorted_items[:5]:
                    st.write(f"**{emo}**: {score:.3f}")
            else:
                st.write("No video emotion output yet.")

    st.markdown("### Latency")
    with st.container(border=True):
        latency = current.get("latency", {})
        if latency:
            cols = st.columns(len(latency))
            for idx, (k, v) in enumerate(latency.items()):
                cols[idx].metric(k, f"{v:.2f}s")
        else:
            st.write("No latency data yet.")

    st.markdown("### Recent turns")
    if st.session_state.completed_turns:
        for turn in reversed(st.session_state.completed_turns[-5:]):
            with st.expander(f"Turn {turn.get('turn_id') or '?'}", expanded=False):
                st.write(f"**User:** {turn.get('transcription') or '-'}")
                st.write(f"**Agent:** {turn.get('agent_reply') or '-'}")
                st.write(f"**Decision:** {turn.get('decision') or '-'}")
                if turn.get("latency"):
                    st.json(turn.get("latency"))
    else:
        st.write("No completed turns yet.")

    st.markdown("### Raw console")
    with st.container(border=True):
        if st.session_state.logs:
            logs_text = html.escape("\n".join(st.session_state.logs))
            st.markdown(
                f'<div class="raw-console-scroll">{logs_text}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.write("No logs yet.")

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# LIGHT AUTO-REFRESH
# ============================================================
if st.session_state.process_running:
    time.sleep(AUTO_REFRESH_SECONDS)
    st.rerun()

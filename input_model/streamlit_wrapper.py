import os
import sys
import time
import queue
import signal
import threading
import subprocess
from pathlib import Path

import streamlit as st


# -----------------------------
# Configuration
# -----------------------------
# Change this to the filename of your existing conversation loop script.
TARGET_SCRIPT = "master_fusion.py"
ROOT_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = ROOT_DIR / TARGET_SCRIPT


# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(
    page_title="Multimodal Agent Monitor",
    page_icon="🤖",
    layout="wide",
)


# -----------------------------
# Session state
# -----------------------------
def init_state():
    defaults = {
        "process": None,
        "reader_thread": None,
        "stderr_thread": None,
        "stdout_queue": queue.Queue(),
        "stderr_queue": queue.Queue(),
        "log_lines": [],
        "process_running": False,
        "last_refresh": time.time(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# -----------------------------
# Helpers
# -----------------------------
def append_log(line: str, source: str = "stdout"):
    if line is None:
        return
    clean = line.rstrip("\n")
    if not clean:
        return

    prefix = ""
    if source == "stderr":
        prefix = "[stderr] "

    st.session_state.log_lines.append(prefix + clean)

    # prevent unbounded growth
    if len(st.session_state.log_lines) > 4000:
        st.session_state.log_lines = st.session_state.log_lines[-4000:]


def pump_queue():
    moved = False

    while True:
        try:
            line = st.session_state.stdout_queue.get_nowait()
            append_log(line, "stdout")
            moved = True
        except queue.Empty:
            break

    while True:
        try:
            line = st.session_state.stderr_queue.get_nowait()
            append_log(line, "stderr")
            moved = True
        except queue.Empty:
            break

    proc = st.session_state.process
    if proc is not None and proc.poll() is not None and st.session_state.process_running:
        st.session_state.process_running = False
        append_log(f"\n[system] Process exited with code {proc.returncode}")
        moved = True

    return moved


def reader(pipe, out_queue):
    try:
        for line in iter(pipe.readline, ""):
            out_queue.put(line)
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def start_process():
    if not SCRIPT_PATH.exists():
        append_log(f"[system] Could not find script: {SCRIPT_PATH}")
        return

    if st.session_state.process_running:
        append_log("[system] Process is already running.")
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

    append_log(f"[system] Started process: {SCRIPT_PATH.name}")


def send_stdin(text: str):
    proc = st.session_state.process
    if proc is None or not st.session_state.process_running:
        append_log("[system] Cannot send input because the process is not running.")
        return

    try:
        proc.stdin.write(text)
        proc.stdin.flush()
        visible = text.replace("\n", "\\n")
        append_log(f"[ui → process] sent: {visible}")
    except Exception as exc:
        append_log(f"[system] Failed to send input: {exc}")


def stop_process():
    proc = st.session_state.process
    if proc is None or not st.session_state.process_running:
        append_log("[system] No running process to stop.")
        return

    try:
        if os.name == "nt":
            proc.terminate()
        else:
            proc.send_signal(signal.SIGTERM)
        append_log("[system] Sent terminate signal.")
    except Exception as exc:
        append_log(f"[system] Failed to stop process cleanly: {exc}")


def send_turn_start():
    # Matches: input("🟢 TURN X | Press [ENTER] to start speaking ...")
    send_stdin("\n")


def send_turn_stop():
    # Matches: input("🛑 Press [ENTER] when you are finished talking...")
    send_stdin("\n")


def send_quit():
    # Matches: type 'q' to quit
    send_stdin("q\n")


def clear_logs():
    st.session_state.log_lines = []


# -----------------------------
# UI
# -----------------------------
st.title("🤖 Multimodal Conversational Agent")
st.caption("A lightweight Streamlit wrapper that mirrors your terminal output into a cleaner UI.")

pump_queue()

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Controls")

    if not SCRIPT_PATH.exists():
        st.error(f"Script not found: {SCRIPT_PATH}")
    else:
        st.success(f"Target script: {SCRIPT_PATH.name}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Launch agent", use_container_width=True):
            start_process()
    with c2:
        if st.button("Stop process", use_container_width=True):
            stop_process()

    st.markdown("### Turn controls")
    st.write("These map directly to the two ENTER prompts in your existing CLI loop.")

    if st.button("Start speaking", use_container_width=True, disabled=not st.session_state.process_running):
        send_turn_start()

    if st.button("Stop speaking", use_container_width=True, disabled=not st.session_state.process_running):
        send_turn_stop()

    if st.button("Quit gracefully", use_container_width=True, disabled=not st.session_state.process_running):
        send_quit()

    st.markdown("### Status")
    st.metric("Process running", "Yes" if st.session_state.process_running else "No")
    st.metric("Log lines", len(st.session_state.log_lines))

    auto_refresh = st.checkbox("Auto-refresh logs", value=True)

    refresh_now = st.button("Refresh logs", use_container_width=True)
    if refresh_now:
        pump_queue()

    if st.button("Clear visible logs", use_container_width=True):
        clear_logs()

    st.markdown("### Notes")
    st.info(
        "This first version keeps your core Python script unchanged. "
        "It simply launches it, streams stdout/stderr into Streamlit, and lets you send ENTER/q to stdin."
    )

with right:
    st.subheader("Live console")

    log_container = st.container(border=True)
    with log_container:
        if st.session_state.log_lines:
            rendered = "\n".join(st.session_state.log_lines)
            st.code(rendered, language="text")
        else:
            st.write("No logs yet. Launch the agent to begin.")


# -----------------------------
# Optional lightweight auto-refresh
# -----------------------------
if auto_refresh and st.session_state.process_running:
    time.sleep(0.7)
    st.rerun()

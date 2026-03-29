#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEBUG_FRAMES="$SCRIPT_DIR/input_model/debug_frames"
AUDIO_RECORDINGS="$SCRIPT_DIR/input_model/audio_recordings"
SESSIONS_DIR="$SCRIPT_DIR/sessions_basic"
# Auto-detect Python: macOS venv uses bin/python, Windows uses Scripts/python.exe
if [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON_EXEC="$SCRIPT_DIR/venv/bin/python"
elif [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_EXEC="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
    PYTHON_EXEC="$SCRIPT_DIR/.venv/Scripts/python.exe"
else
    echo "ERROR: Could not find Python in venv or .venv"
    exit 1
fi

# Generate a random participant ID (8 hex chars)
PARTICIPANT_ID=$(openssl rand -hex 4)

echo "=== Session Runner (Basic) ==="
echo "Participant ID    : $PARTICIPANT_ID"
echo "Debug frames      : $DEBUG_FRAMES"
echo "Audio recordings  : $AUDIO_RECORDINGS"

# 1. Remove old debug_frames and audio_recordings
if [ -d "$DEBUG_FRAMES" ]; then
    echo "Removing old debug_frames..."
    rm -rf "$DEBUG_FRAMES"
fi

if [ -d "$AUDIO_RECORDINGS" ]; then
    echo "Removing old audio_recordings..."
    rm -rf "$AUDIO_RECORDINGS"
fi

# 2. Create session folder upfront so the log can be written during the run
SAVE_PATH="$SESSIONS_DIR/$PARTICIPANT_ID"
mkdir -p "$SAVE_PATH"
LOG_FILE="$SAVE_PATH/session.log"

# 3. Run master_fusion_basic.py, capturing stdout+stderr to session.log while still printing to terminal
echo "Starting master_fusion_basic.py..."
cd "$SCRIPT_DIR/input_model"
if [ "$(uname -s)" = "Darwin" ]; then
    script -q "$LOG_FILE" $PYTHON_EXEC -u master_fusion_basic.py
elif command -v script >/dev/null 2>&1; then
    # util-linux script syntax: -c runs command while recording the terminal session.
    script -q "$LOG_FILE" -c "$PYTHON_EXEC -u master_fusion_basic.py"
else
    echo "Notice: 'script' command not found. Falling back to tee logging."
    "$PYTHON_EXEC" -u master_fusion_basic.py 2>&1 | tee "$LOG_FILE"
fi

# 4. Save remaining session artifacts

if [ -d "$DEBUG_FRAMES" ]; then
    echo "Saving debug_frames to $SAVE_PATH/debug_frames ..."
    cp -r "$DEBUG_FRAMES" "$SAVE_PATH/debug_frames"
else
    echo "Warning: debug_frames not found after session — nothing to save."
fi

if [ -d "$AUDIO_RECORDINGS" ]; then
    echo "Saving audio_recordings to $SAVE_PATH/audio_recordings ..."
    cp -r "$AUDIO_RECORDINGS" "$SAVE_PATH/audio_recordings"
else
    echo "Warning: audio_recordings not found after session — nothing to save."
fi

echo "Session saved → $SAVE_PATH"

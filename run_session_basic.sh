#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEBUG_FRAMES="$SCRIPT_DIR/input_model/debug_frames"
AUDIO_RECORDINGS="$SCRIPT_DIR/input_model/audio_recordings"
SESSIONS_DIR="$SCRIPT_DIR/sessions_basic"

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

# 3. Run master_fusion_basic.py, capturing stdout+stderr to session.log while still printing to terminal
echo "Starting master_fusion_basic.py..."
cd "$SCRIPT_DIR/input_model"
script -q "$SAVE_PATH/session.log" python3 -u master_fusion_basic.py

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

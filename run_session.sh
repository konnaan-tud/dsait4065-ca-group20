#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHROMA_DB="$SCRIPT_DIR/input_model/chroma_db"
DEBUG_FRAMES="$SCRIPT_DIR/input_model/debug_frames"
AUDIO_RECORDINGS="$SCRIPT_DIR/input_model/audio_recordings"
SESSIONS_DIR="$SCRIPT_DIR/sessions"

# Generate a random participant ID (8 hex chars)
PARTICIPANT_ID=$(openssl rand -hex 4)

echo "=== Session Runner ==="
echo "Participant ID    : $PARTICIPANT_ID"
echo "ChromaDB path     : $CHROMA_DB"
echo "Debug frames      : $DEBUG_FRAMES"
echo "Audio recordings  : $AUDIO_RECORDINGS"

# 1. Remove old chroma_db, debug_frames, and audio_recordings
if [ -d "$CHROMA_DB" ]; then
    echo "Removing old chroma_db..."
    rm -rf "$CHROMA_DB"
fi

if [ -d "$DEBUG_FRAMES" ]; then
    echo "Removing old debug_frames..."
    rm -rf "$DEBUG_FRAMES"
fi

if [ -d "$AUDIO_RECORDINGS" ]; then
    echo "Removing old audio_recordings..."
    rm -rf "$AUDIO_RECORDINGS"
fi

# 2. Run master_fusion_basic.py from its directory
echo "Starting master_fusion_basic.py..."
cd "$SCRIPT_DIR/input_model"
"$SCRIPT_DIR/.venv/bin/python" master_fusion.py

# 3. Save session artifacts under sessions/<participant_id>/
SAVE_PATH="$SESSIONS_DIR/$PARTICIPANT_ID"
mkdir -p "$SAVE_PATH"

if [ -d "$CHROMA_DB" ]; then
    echo "Saving chroma_db state to $SAVE_PATH/chroma_db ..."
    cp -r "$CHROMA_DB" "$SAVE_PATH/chroma_db"
else
    echo "Warning: chroma_db not found after session — nothing to save."
fi

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

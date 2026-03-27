#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHROMA_DB="$SCRIPT_DIR/chroma_db"
SESSIONS_DIR="$SCRIPT_DIR/sessions"

# Generate a random participant ID (8 hex chars)
PARTICIPANT_ID=$(openssl rand -hex 4)

echo "=== Session Runner ==="
echo "Participant ID : $PARTICIPANT_ID"
echo "ChromaDB path  : $CHROMA_DB"

# 1. Remove old chroma_db
if [ -d "$CHROMA_DB" ]; then
    echo "Removing old chroma_db..."
    rm -rf "$CHROMA_DB"
fi

# 2. Run master_fusion_basic.py from its directory
echo "Starting master_fusion_basic.py..."
cd "$SCRIPT_DIR/input_model"
"$SCRIPT_DIR/.venv/bin/python" master_fusion_basic.py

# 3. Save the db state under sessions/<participant_id>/
SAVE_PATH="$SESSIONS_DIR/$PARTICIPANT_ID"
mkdir -p "$SAVE_PATH"

if [ -d "$CHROMA_DB" ]; then
    echo "Saving chroma_db state to $SAVE_PATH/chroma_db ..."
    cp -r "$CHROMA_DB" "$SAVE_PATH/chroma_db"
    echo "Session saved → $SAVE_PATH"
else
    echo "Warning: chroma_db not found after session — nothing to save."
fi

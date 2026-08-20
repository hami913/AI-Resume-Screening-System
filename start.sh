#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$PROJECT_DIR" || exit 1

echo "=========================================="
echo "   AI Career Assistant Starting..."
echo "=========================================="

# Llama is optional.
# If LLAMA_SERVER is available and the model exists,
# start it automatically. Otherwise Streamlit starts normally.

LLAMA_PID=""

LLAMA_SERVER="${LLAMA_SERVER:-}"

MODEL="$PROJECT_DIR/models/gguf/llama-3.1-8b-instruct-q4_k_m.gguf"
LORA="$PROJECT_DIR/models/gguf/llama-3.1-8b-career-assistant-lora-f16.gguf"

if [ -n "$LLAMA_SERVER" ] && \
   [ -x "$LLAMA_SERVER" ] && \
   [ -f "$MODEL" ]; then

    echo "[1/2] Starting Llama server..."

    LLAMA_CMD=(
        "$LLAMA_SERVER"
        -m "$MODEL"
        --host 127.0.0.1
        --port 8080
    )

    if [ -f "$LORA" ]; then
        LLAMA_CMD+=(--lora "$LORA")
    fi

    "${LLAMA_CMD[@]}" > llama-server.log 2>&1 &

    LLAMA_PID=$!

    echo "Llama server PID: $LLAMA_PID"
    echo "Waiting for Llama server..."

    for i in {1..60}; do
        if curl -s http://127.0.0.1:8080/v1/models > /dev/null 2>&1; then
            echo "Llama server is ready!"
            break
        fi

        if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
            echo "WARNING: Llama server stopped."
            echo "Continuing without Llama..."
            LLAMA_PID=""
            break
        fi

        sleep 1
    done

else
    echo "[1/2] Llama server not configured."
    echo "Starting Streamlit without local Llama..."
fi

echo "[2/2] Starting Streamlit..."

if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
elif [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

streamlit run app.py

echo ""
echo "Streamlit stopped."

if [ -n "$LLAMA_PID" ]; then
    echo "Stopping Llama server..."
    kill "$LLAMA_PID" 2>/dev/null
fi

echo "AI Career Assistant stopped."

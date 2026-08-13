#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"

MODEL="$PROJECT_DIR/models/gguf/llama-3.1-8b-instruct-q4_k_m.gguf"
LORA="$PROJECT_DIR/models/gguf/llama-3.1-8b-career-assistant-lora-f16.gguf"

cd "$PROJECT_DIR" || exit 1

echo "=========================================="
echo "   AI Career Assistant Starting..."
echo "=========================================="

echo "[1/2] Starting Llama server..."

"$LLAMA_SERVER" \
  -m "$MODEL" \
  --lora "$LORA" \
  --host 127.0.0.1 \
  --port 8080 > llama-server.log 2>&1 &

LLAMA_PID=$!

echo "Llama server PID: $LLAMA_PID"
echo "Waiting for Llama server..."

for i in {1..60}; do
    if curl -s http://127.0.0.1:8080/v1/models > /dev/null 2>&1; then
        echo "Llama server is ready!"
        break
    fi

    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "ERROR: Llama server stopped."
        echo "Check llama-server.log"
        exit 1
    fi

    sleep 1
done

if ! curl -s http://127.0.0.1:8080/v1/models > /dev/null 2>&1; then
    echo "ERROR: Llama server did not start."
    echo "Check llama-server.log"
    exit 1
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
echo "Stopping Llama server..."

kill "$LLAMA_PID" 2>/dev/null

echo "AI Career Assistant stopped."

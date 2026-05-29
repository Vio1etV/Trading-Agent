#!/usr/bin/env bash
# One command to start the demo.
#
# From the project root, run:
#     bash run_demo.sh
#
# It will:
#   1. Kill any old streamlit / cloudflared
#   2. Start streamlit in the background (logs at /tmp/streamlit.log)
#   3. Wait for it to be ready
#   4. Start cloudflared in the foreground and print the public URL
#
# Ctrl+C stops cloudflared (the demo URL goes down but streamlit keeps running).

set -e

cd "$(dirname "$0")"

echo "==> killing old streamlit / cloudflared..."
pkill -9 -f streamlit 2>/dev/null || true
pkill -9 -f cloudflared 2>/dev/null || true
sleep 1

echo "==> starting streamlit on port 6006 (background)..."
nohup streamlit run app.py \
    --server.port 6006 \
    --server.address 0.0.0.0 \
    > /tmp/streamlit.log 2>&1 &

echo "==> waiting for streamlit to be ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:6006 >/dev/null 2>&1; then
        echo "    streamlit is up."
        break
    fi
    sleep 1
done

CLOUDFLARED=/tmp/cloudflared
if [ ! -x "$CLOUDFLARED" ]; then
    echo "==> cloudflared not found, downloading..."
    wget -q \
        https://gh-proxy.com/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
        -O "$CLOUDFLARED"
    chmod +x "$CLOUDFLARED"
fi

echo ""
echo "============================================================"
echo "==> Public URL will appear below — copy it into your browser."
echo "==> Ctrl+C to stop the tunnel."
echo "============================================================"
echo ""

"$CLOUDFLARED" tunnel --url http://localhost:6006 --protocol http2

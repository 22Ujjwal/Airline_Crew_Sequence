#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/aa_env"

echo "=== Starting AA Crew Risk API (port 8000) ==="
"$VENV/bin/uvicorn" src.api:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

echo "=== Starting Next.js dev server (port 3000) ==="
cd "$ROOT/web" && npm run dev &
WEB_PID=$!

echo ""
echo "  API:  http://localhost:8000/api/health"
echo "  App:  http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers."

cleanup() {
  kill $API_PID $WEB_PID 2>/dev/null
  exit 0
}
trap cleanup INT TERM
wait

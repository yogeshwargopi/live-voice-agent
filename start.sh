#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required."
  exit 1
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env. Paste your OpenAI API key into .env, then run ./start.sh again."
  exit 1
fi

if ! grep -E -q 'OPENAI_API_KEY=.+' .env; then
  echo "Paste your OpenAI API key into .env, then run ./start.sh again."
  exit 1
fi

exec python app.py

#!/bin/zsh
# Jalankan web UI transkrip di http://127.0.0.1:5100
set -e
cd "$(dirname "$0")"
# python@3.12 Homebrew di Mac ini perlu expat Homebrew agar pip/pyexpat jalan.
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib
if [ ! -d .venv ]; then
  /opt/homebrew/bin/python3.12 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
exec .venv/bin/python app.py

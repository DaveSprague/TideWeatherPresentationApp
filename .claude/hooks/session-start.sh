#!/bin/bash
# Session start hook for TideWeatherPresentationApp
# Installs Python dependencies so the Dash app and all imports work.
# Only runs in Claude Code web sessions.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo "Installing Python dependencies..."
pip install -q --ignore-installed -r "$CLAUDE_PROJECT_DIR/requirements.txt"
pip install -q python-dotenv

# Make sure the project root is on PYTHONPATH so `import app` and
# `from presentation_app...` both resolve without a venv.
echo "export PYTHONPATH=\"$CLAUDE_PROJECT_DIR:\${PYTHONPATH:-}\"" >> "$CLAUDE_ENV_FILE"

echo "Dependencies installed."

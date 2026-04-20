#!/bin/bash
# Rescue terminal — launched by ttyd on port 7681.
cd /workspace

# Load Anthropic API key if user has saved one
if [ -f /workspace/.anthropic_key ]; then
    export ANTHROPIC_API_KEY="$(cat /workspace/.anthropic_key)"
fi

# Skip the npm-registry version check on every launch (shaves seconds off startup).
export DISABLE_AUTOUPDATER=1

cat <<'BANNER'
================================================================
  🛟  Rescue Terminal — /workspace on vast instance 35313221
================================================================
  Starting Claude Code. On first run it will prompt for login
  (or set ANTHROPIC_API_KEY and save it to /workspace/.anthropic_key
   before hitting the rescue button next time).

  Useful paths:
    /workspace/ComfyUI         ComfyUI install
    /workspace/studio.py       Studio source
    /workspace/comfy.log       ComfyUI log
    /workspace/studio.log      Studio log
  tmux sessions: comfyui, studio, dl, ssh_tmux

  Press Ctrl-D twice to exit claude → drop to bash shell.
================================================================
BANNER

claude || true

echo
echo "[claude exited — dropped to bash. Ctrl-D to close tab.]"
exec bash

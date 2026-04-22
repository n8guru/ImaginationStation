#!/bin/bash
# Launch ComfyUI, Studio, and the ttyd rescue terminal in tmux sessions.
# Idempotent: kills existing sessions first.
#
# `env -u TS_AUTHKEY` strips the Tailscale auth key from the subprocess env
# so it doesn't reach the studio LLM (which has run_shell) or the rescue
# terminal's Claude Code. The key only needs to live long enough for
# `tailscale up` in bootstrap.sh.
set -u

for s in comfyui studio rescue; do
    tmux has-session -t "$s" 2>/dev/null && tmux kill-session -t "$s"
done

env -u TS_AUTHKEY tmux new-session -d -s comfyui \
    'cd /workspace/ComfyUI && source venv/bin/activate && \
     python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header \
     --extra-model-paths-config /workspace/ComfyUI/extra_model_paths.yaml \
     2>&1 | tee /workspace/comfy.log'

sleep 3

env -u TS_AUTHKEY tmux new-session -d -s studio \
    '/workspace/ComfyUI/venv/bin/python /workspace/studio.py 2>&1 | tee /workspace/studio.log'

env -u TS_AUTHKEY tmux new-session -d -s rescue \
    'ttyd -W -p 7681 -t fontSize=14 -t "theme={\"background\":\"#1e1e1e\"}" \
     bash /workspace/rescue.sh'

echo "started: comfyui(8188)  studio(3000)  rescue(7681)"

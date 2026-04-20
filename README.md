# ImaginationStation

Portable ComfyUI studio for vast.ai: LLM chat + ComfyUI canvas in one browser tab,
with a rescue terminal running Claude Code on the side.

## Ports

| Port | Service                                              |
|-----:|------------------------------------------------------|
| 3000 | Studio UI (chat + embedded ComfyUI iframe)           |
| 8188 | ComfyUI (direct)                                     |
| 7681 | Rescue terminal (ttyd → Claude Code CLI)             |

## One-shot vast.ai onstart

Paste this into the instance's "On-start Script" field:

```bash
#!/bin/bash
set -e
cd /root
[ -d ImaginationStation ] || git clone https://github.com/n8guru/ImaginationStation.git
cd ImaginationStation
git pull --ff-only || true
bash bootstrap.sh
```

That's the whole thing. `bootstrap.sh` is idempotent: it installs OS packages,
sets up the ComfyUI venv, installs custom nodes, downloads starter models,
and launches the three tmux services.

## Restarting on a live instance

```bash
/workspace/start_all.sh
```

## Editing the studio

`studio.py` and `rescue.sh` are symlinked from `/workspace/` into this repo
by `bootstrap.sh`. Edit them in place, then `git commit && git push`.
Next instance you spin up picks up the changes automatically.

## Files

- `bootstrap.sh`  — full provisioning (OS deps, ComfyUI, venv, models, launch)
- `start_all.sh`  — tmux launcher for comfyui/studio/rescue
- `studio.py`     — Gradio UI: LLM chat with tool-calling into ComfyUI, iframe on the right
- `rescue.sh`     — ttyd → Claude Code CLI entry point

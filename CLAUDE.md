# ImaginationStation — Claude Code orientation

You are running in the **rescue terminal** of a vast.ai GPU instance that was
provisioned by `bootstrap.sh` from this repo. The user hit the "rescue" button
because something about the studio's autonomous director needs human/Claude
judgment. Orient yourself before acting.

## The three services

| Port | Service | tmux session | What it is |
|-----:|---------|--------------|------------|
| 8188 | ComfyUI | `comfyui` | Image/video generation engine. API at `http://127.0.0.1:8188`. |
| 3000 | Studio  | `studio`  | Gradio UI: chat with an OpenRouter LLM ("director") that has tool-calling into ComfyUI + `run_shell`. See `studio.py`. |
| 7681 | Rescue  | `rescue`  | ttyd → `claude` (this terminal you're in). |

Logs: `/workspace/{comfy,studio}.log`. Restart everything: `bash /workspace/start_all.sh`.

## Model library architecture — **read this before touching models**

Models live in **DO Spaces** bucket `imagination-video` (sfo3), organized as:

```
s3://imagination-video/
├── models/{checkpoints,loras,vae,controlnet,...}/<filename>
├── previews/<sha256>.jpg
└── library/manifest.db          ← SQLite, source of truth
```

The manifest (`/workspace/library/manifest.db` locally) has one row per model
with sha256, display name, base model, trigger words, spaces_key, etc. It's
pulled from Spaces on boot (`bootstrap.sh`) and pushed back after every ingest.

**Concurrency**: V1 is single-writer. Do NOT run `ingest.py` from two boxes
simultaneously — the SQLite file is synced whole and concurrent writes clobber.

### The director's model-sourcing flow (enforced in its system prompt)

1. `search_library(query=...)` — check manifest first.
2. If hit: `pull_model(filename=...)` — fast s5cmd pull from Spaces.
3. If miss: `install_model(url, dest_type)` — downloads from source with
   auto-applied HF/Civitai auth, then auto-ingests (upload to Spaces + manifest
   row). **Never call `run_shell` to wget a model directly** — that bypasses
   backup and leaves orphan files.

### Adding a model manually (from this terminal)

```bash
cd /root/ImaginationStation
python ingest.py civitai 288982                  # by CivitAI model ID
python ingest.py hf org/repo/file.safetensors    # by HF repo path
python ingest.py local ./file.safetensors --base SDXL --category lora \
       --triggers "style keyword" --name "Display Name"
```

All three paths: SHA-256 dedupe → upload to Spaces → write manifest row → push.

## Credentials (all in env on this box)

- `HF_TOKEN` — HuggingFace (auto-used by `install_model` for HF URLs)
- `CIVITAI_API_TOKEN` — CivitAI (ditto)
- `COMFY_S3_{ENDPOINT,BUCKET,ACCESS_KEY,SECRET_KEY,REGION}` — DO Spaces
- `OPENROUTER_KEY` — saved to `/workspace/.openrouter_key` after first use
- `TS_AUTHKEY` — scrubbed by bootstrap.sh after `tailscale up`. If you see it
  still set in the studio LLM's env, that's a regression — check `start_all.sh`.

## Peer access — isolation by design

By design, the studio LLM and this rescue claude should have **NO** access to:
- `jobscout` (DO droplet, 100.86.135.77)
- `n8razer` (storage host, 100.102.77.86)

If you find a way to reach either (SOCKS5 through `localhost:1055`, direct IP,
port-scan, etc.), **that's a security hole to report, not a feature to use**.
Tailscale ACLs should block it at the network level, not rely on service auth.

## Editing the code

All source files in `/workspace/` are **symlinks into `/root/ImaginationStation/`**,
which is a git checkout of https://github.com/n8guru/ImaginationStation.

```bash
cd /root/ImaginationStation
# edit studio.py, bootstrap.sh, ingest.py, etc.
git add -A && git commit -m "..." && git push
```

Next spawn pulls the changes via `bootstrap.sh` (which does `git pull --ff-only`).
Restart the studio to pick up changes on THIS box: `bash /workspace/start_all.sh`.

## Common gotchas

- `wget -O <dest>` truncates the destination *before* the HTTP response. A
  failed download leaves a 0-byte stub. `install_model` now downloads to
  `<dest>.part` and atomic-renames on success — do the same if you write any
  new download code.
- `studio.py` is imported from `/workspace/studio.py` (a symlink), so
  `Path(__file__).parent` resolves to `/workspace` — use `.resolve().parent`
  to get the repo dir for `sys.path` / library imports.
- The bucket name is `imagination-video` (historical; it holds models AND
  outputs AND the manifest). Don't rename without a migration.
- S3 `NoSuchKey` on first-ever provision is normal — `bootstrap.sh` falls back
  to `init_db()` to create an empty manifest.

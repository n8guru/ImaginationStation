# ImaginationStation — Claude Code orientation

You are running on a **vast.ai GPU instance** provisioned by `bootstrap.sh`.
This box is ephemeral — it will be destroyed when the session ends. Your job
is to help with ComfyUI generation, model management, and studio issues.

## Architecture overview

This GPU instance is part of the **Forage Imagination** system:

- **Droplet** (forage.ink) — runs the director LLM for freestyle sessions,
  manages session logs, serves the cockpit UI at imagination.forage.ink
- **This GPU instance** — runs ComfyUI + Studio (Gradio chat with DeepSeek).
  The director on the droplet can call `request_image` which POSTs to this
  box's `:3000/api/generate` endpoint
- **DO Spaces** (`imagination-models` bucket, sfo3) — permanent model library.
  All models, LoRAs, VAEs, previews, and the manifest.db live here. Synced
  to this box at boot via `s5cmd`
- **N8Razer** — Nate's local GPU machine. Has ComfyUI + storage. Not directly
  accessible from this box (Tailscale ACLs)

## The three services

| Port | Service | tmux session | What it is |
|-----:|---------|--------------|------------|
| 8188 | ComfyUI | `comfyui` | Image/video generation engine. API at `http://127.0.0.1:8188` |
| 3000 | Studio  | `studio`  | Gradio UI: DeepSeek chat with tool-calling into ComfyUI + shell. See `studio.py` |
| 7681 | Rescue  | `rescue`  | ttyd → `claude` (this terminal) |

Logs: `/workspace/{comfy,studio}.log`. Restart everything: `bash /workspace/start_all.sh`.

## Model library — **read before touching models**

Models live in **DO Spaces** bucket `imagination-models` (sfo3):

```
s3://imagination-models/
├── models/{checkpoints,loras,vae,controlnet,...}/<filename>
├── previews/<sha256>.jpg
└── library/manifest.db          ← SQLite, source of truth
```

The manifest (`/workspace/library/manifest.db` locally) has one row per model
with sha256, display name, base model, trigger words, spaces_key, etc. It's
pulled from Spaces on boot and pushed back after every ingest.

**Current library**: ~52 models (15 checkpoints, 36 LoRAs, 2 VAEs, 1 upscaler).
Includes NSFW models from CivitAI. Use `search_library` in the studio to browse.

**On-demand pull**: Models are NOT synced at boot — only the manifest is pulled.
The director uses `search_library` to find models, then `pull_model(filename)`
to download just the ones needed for the current session. This keeps boot fast
and avoids pulling 90+ GB when only a few models are needed.

**Concurrency**: V1 is single-writer. Do NOT run `ingest.py` from two boxes
simultaneously — the SQLite manifest is synced whole and concurrent writes clobber.

### The director's model-sourcing flow (enforced in system prompt)

1. `search_library(query=...)` — check manifest first
2. If hit: `pull_model(filename=...)` — fast s5cmd pull from Spaces
3. If miss: `install_model(url, dest_type)` — downloads from source with
   HF/Civitai auth, then auto-ingests (upload to Spaces + manifest row)
4. **Never `run_shell` + wget a model** — bypasses backup, leaves orphans

### Image review loop (enforced in system prompt)

After generating an image, DeepSeek should call `review_image(image_path, focus)`
which sends the image to Grok 4.1 Fast via OpenRouter for quality review.
Returns structured verdict: pass/flag/fail with rating 1-10 and prompt revision
notes. If rating < 7, iterate (up to 3 times).

### Adding a model manually

```bash
cd /root/ImaginationStation
python ingest.py civitai 288982                  # by CivitAI model ID
python ingest.py hf org/repo/file.safetensors    # by HF repo path
python ingest.py local ./file.safetensors --base SDXL --category lora \
       --triggers "style keyword" --name "Display Name"
```

All paths: SHA-256 dedupe → upload to Spaces → manifest row → push manifest.

## Credentials (env vars on this box)

- `HF_TOKEN` — HuggingFace (used by `install_model` for HF URLs)
- `CIVITAI_API_TOKEN` — CivitAI (ditto)
- `COMFY_S3_{ENDPOINT,BUCKET,ACCESS_KEY,SECRET_KEY,REGION}` — DO Spaces
- OpenRouter key saved to `/workspace/.openrouter_key`
- `TS_AUTHKEY` — scrubbed by bootstrap.sh after `tailscale up`

## Session persistence — CRITICAL

This box is **ephemeral**. When it gets destroyed, anything not saved is lost.

**What persists automatically:**
- Model files + manifest → DO Spaces (via ingest.py auto-upload)
- Code changes → GitHub (`n8guru/ImaginationStation`) if you commit + push

**What you must save manually before shutdown:**
- Any changes to `studio.py`, `CLAUDE.md`, `bootstrap.sh` → `git commit && git push`
- Generated outputs worth keeping → download to session artifacts on droplet
  or sync to N8Razer via the "Save to N8Razer" button in the cockpit UI
- Session learnings → update CLAUDE.md in the repo so next spawn benefits

**Before shutdown checklist:**
1. `cd /root/ImaginationStation && git add -A && git diff --cached --stat`
2. If changes: `git commit -m "..." && git push`
3. Verify models ingested: `python -c "from library.manifest import open_db; c=open_db(); print(c.execute('SELECT COUNT(*) FROM models').fetchone())"`
4. Push manifest: happens automatically during ingest, but verify with
   `s5cmd --endpoint-url $COMFY_S3_ENDPOINT cp /workspace/library/manifest.db s3://$COMFY_S3_BUCKET/library/manifest.db`

## Peer access — isolation by design

This box should have **NO** access to:
- `jobscout` (DO droplet) — except via the imagination API callback
- `n8razer` (storage host) — no access at all

If you can reach either via SSH, SOCKS5, or direct IP, that's a security hole
to report, not a feature to use.

## Editing the code

Source files in `/workspace/` are **symlinks into `/root/ImaginationStation/`**.

```bash
cd /root/ImaginationStation
# edit, then:
git add -A && git commit -m "..." && git push
bash /workspace/start_all.sh  # restart to pick up changes on this box
```

Next spawn gets changes via `git clone` in bootstrap.sh.

## Common gotchas

- `install_model` downloads to `<dest>.part` and atomic-renames on success.
  Never use `wget -O <dest>` directly — failed downloads leave 0-byte stubs.
- `studio.py` is a symlink from `/workspace/` → use `Path(__file__).resolve().parent`
  for repo-relative imports.
- The bucket is `imagination-models` (not `imagination-video`).
- S3 `NoSuchKey` on first provision is normal — bootstrap falls back to `init_db()`.
- GPU instances lack `/dev/net/tun` — Tailscale runs in userspace networking mode.

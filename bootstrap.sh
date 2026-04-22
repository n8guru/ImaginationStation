#!/bin/bash
# ImaginationStation bootstrap — idempotent provisioning for a vast.ai instance.
# Assumes the repo is already cloned to /workspace/ImaginationStation or /root/ImaginationStation.
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[bootstrap]${NC} $*"; }
warn() { echo -e "${YELLOW}[bootstrap]${NC} $*"; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE=/workspace
COMFY="$WORKSPACE/ComfyUI"

# 0. Tailscale — join tailnet if auth key provided
if [ -n "${TS_AUTHKEY:-}" ]; then
    log "installing Tailscale"
    curl -fsSL https://tailscale.com/install.sh | sh

    # Start tailscaled manually — Docker containers lack systemd and /dev/net/tun.
    # Use userspace networking (SOCKS proxy mode) if TUN device is unavailable.
    log "starting tailscaled"
    mkdir -p /var/lib/tailscale /var/run/tailscale
    if [ -e /dev/net/tun ]; then
        tailscaled --state=/var/lib/tailscale/tailscaled.state \
                   --socket=/var/run/tailscale/tailscaled.sock &
    else
        warn "/dev/net/tun not available — using userspace networking"
        tailscaled --state=/var/lib/tailscale/tailscaled.state \
                   --socket=/var/run/tailscale/tailscaled.sock \
                   --tun=userspace-networking --socks5-server=localhost:1055 &
    fi
    sleep 3

    log "joining tailnet as ${TS_HOSTNAME:-imagination}"
    if tailscale up --authkey="$TS_AUTHKEY" --hostname="${TS_HOSTNAME:-imagination}" --accept-routes 2>&1; then
        TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "pending")
        log "Tailscale up — IP: $TAILSCALE_IP"
    else
        warn "Tailscale failed to join — continuing without tailnet (services still accessible via public IP)"
    fi

    # Scrub the auth key once we're joined — it's reusable and must not leak to
    # subprocess env (the studio LLM has shell access). Unset here + in future
    # interactive shells. Tmux sessions launched by start_all.sh use `env -u`.
    unset TS_AUTHKEY
    if ! grep -q "unset TS_AUTHKEY" /root/.bashrc 2>/dev/null; then
        echo "unset TS_AUTHKEY  # scrubbed by ImaginationStation/bootstrap.sh" >> /root/.bashrc
    fi
fi

# 1. OS packages
log "apt packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -yqq wget curl git build-essential libgl1-mesa-glx libglib2.0-0 \
    tmux screen ffmpeg htop python3 python3-pip python3-venv ttyd

# Node (for Claude Code CLI) — only if missing
if ! command -v node >/dev/null; then
    log "installing Node 20 (for Claude Code CLI)"
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -yqq nodejs
fi
if ! command -v claude >/dev/null; then
    log "installing Claude Code CLI"
    npm install -g @anthropic-ai/claude-code
fi

# 2. ComfyUI
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"
if [ ! -d "$COMFY/.git" ]; then
    log "cloning ComfyUI"
    git clone https://github.com/comfyanonymous/ComfyUI.git
else
    log "updating ComfyUI"
    (cd "$COMFY" && git pull --ff-only || true)
fi

cd "$COMFY"
if [ ! -d venv ]; then
    log "creating venv"
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install gradio openai requests python-dotenv

# 3. ComfyUI-Manager + key custom nodes
mkdir -p custom_nodes
cd custom_nodes
KEY_NODES=(
    "https://github.com/ltdrdata/ComfyUI-Manager.git"
    "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git"
    "https://github.com/Fannovel16/comfyui_controlnet_aux.git"
    "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git"
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git"
)
for url in "${KEY_NODES[@]}"; do
    dir=$(basename "$url" .git)
    if [ ! -d "$dir" ]; then
        log "installing custom node: $dir"
        git clone "$url"
        [ -f "$dir/requirements.txt" ] && pip install -r "$dir/requirements.txt" || true
    fi
done
cd "$COMFY"

# 4. Models — sync from DO Spaces library (managed by ingest.py)
mkdir -p models/{checkpoints,loras,controlnet,vae,text_encoders,flux,embeddings,upscalers}

# Install s5cmd for fast parallel S3 sync
if ! command -v s5cmd >/dev/null; then
    log "installing s5cmd"
    curl -fsSL -o /tmp/s5cmd.deb "https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_linux_amd64.deb"
    dpkg -i /tmp/s5cmd.deb
    rm -f /tmp/s5cmd.deb
fi

# Sync models from Spaces if S3 creds are available
if [ -n "${COMFY_S3_ACCESS_KEY:-}" ] && [ -n "${COMFY_S3_SECRET_KEY:-}" ]; then
    BUCKET="${COMFY_S3_BUCKET:-imagination-models}"
    ENDPOINT="${COMFY_S3_ENDPOINT:-https://sfo3.digitaloceanspaces.com}"

    export AWS_ACCESS_KEY_ID="$COMFY_S3_ACCESS_KEY"
    export AWS_SECRET_ACCESS_KEY="$COMFY_S3_SECRET_KEY"
    S5="s5cmd --endpoint-url $ENDPOINT"

    log "syncing models from s3://$BUCKET/models/ ..."
    $S5 sync "s3://${BUCKET}/models/*" models/ || warn "model sync failed (bucket may be empty)"

    # Pull manifest — or initialize an empty one if Spaces doesn't have it yet.
    # Either way, /workspace/library/manifest.db must exist so studio.py's
    # search_library / list_checkpoints return proper results (even if empty)
    # instead of {"error": "Manifest not available"} — which was pushing the
    # director to fall back to raw wget.
    mkdir -p /workspace/library
    if ! $S5 cp "s3://${BUCKET}/library/manifest.db" /workspace/library/manifest.db 2>/dev/null; then
        warn "no manifest.db on Spaces yet — initializing empty one"
        "$COMFY/venv/bin/python" -c "
import sys; sys.path.insert(0, '$REPO_DIR')
from library.manifest import init_db
init_db().close()
print('initialized empty manifest.db')
" || warn "manifest init failed (check boto3/library install)"
    fi

    # Reconcile: log orphan files (in models/ but not in manifest)
    if [ -f /workspace/library/manifest.db ]; then
        log "reconciling models against manifest..."
        cd "$COMFY"
        "$COMFY/venv/bin/python" -c "
import sys
sys.path.insert(0, '$REPO_DIR')
from library.manifest import open_db, reconcile
conn = open_db('/workspace/library/manifest.db')
result = reconcile(conn, '$COMFY/models')
conn.close()
if result['orphans']:
    print(f'WARNING: {len(result[\"orphans\"])} orphan files in models/ without manifest rows:')
    for o in result['orphans']:
        print(f'  {o[\"spaces_key\"]} ({o[\"size\"]} bytes)')
    print('These files were not ingested through the pipeline.')
    print('Run: python ingest.py local <file> --base <model> --category <cat>')
else:
    print('All model files have manifest entries.')
if result['missing']:
    print(f'Note: {len(result[\"missing\"])} manifest entries without local files (may still be syncing).')
" 2>&1 || warn "reconciliation check failed (non-fatal)"
    fi

    unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
else
    warn "No S3 credentials — skipping model sync. Models dir will be empty."
    warn "Set COMFY_S3_ACCESS_KEY and COMFY_S3_SECRET_KEY to enable library sync."
fi

cat > extra_model_paths.yaml << 'EOF'
default:
  base_path: /workspace/ComfyUI/models
  checkpoints: checkpoints
  loras: loras
  controlnet: controlnet
  vae: vae
  text_encoders: text_encoders
  flux: flux
  embeddings: embeddings
  upscalers: upscalers
EOF

# 5. Link in studio files from the repo
log "linking studio files"
ln -sf "$REPO_DIR/studio.py"      "$WORKSPACE/studio.py"
ln -sf "$REPO_DIR/rescue.sh"      "$WORKSPACE/rescue.sh"
ln -sf "$REPO_DIR/start_all.sh"   "$WORKSPACE/start_all.sh"
ln -sf "$REPO_DIR/CLAUDE.md"      "$WORKSPACE/CLAUDE.md"
chmod +x "$REPO_DIR/rescue.sh" "$REPO_DIR/start_all.sh" "$REPO_DIR/bootstrap.sh"

# 6. Launch
log "starting services"
bash "$WORKSPACE/start_all.sh"

PUBLIC_IP=$(curl -fsS --max-time 5 ifconfig.me || echo "<instance-ip>")
TS_URL=""
if command -v tailscale >/dev/null && tailscale status >/dev/null 2>&1; then
    TS_HOST=$(tailscale status --json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['Self']['DNSName'].rstrip('.'))" 2>/dev/null || echo "")
    if [ -n "$TS_HOST" ]; then
        TS_URL="  🔒 Tailscale Studio:        http://${TS_HOST}:3000
  🔒 Tailscale ComfyUI:       http://${TS_HOST}:8188
  🔒 Tailscale Rescue:        http://${TS_HOST}:7681"
    fi
fi
cat <<DONE

===============================================================
  ImaginationStation ready
---------------------------------------------------------------
  🎨 Studio (chat + ComfyUI):  http://${PUBLIC_IP}:3000
  🖼  ComfyUI direct:           http://${PUBLIC_IP}:8188
  🛟 Rescue terminal (Claude): http://${PUBLIC_IP}:7681
${TS_URL:+---------------------------------------------------------------
$TS_URL}
---------------------------------------------------------------
  tmux sessions: comfyui, studio, rescue
  Logs: /workspace/{comfy,studio}.log
  Restart: /workspace/start_all.sh
===============================================================
DONE

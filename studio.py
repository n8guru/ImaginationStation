"""
ComfyUI Studio — LLM chat + ComfyUI iframe on one page.
LLM (via OpenRouter) has tool-calling access to ComfyUI API + shell.
"""
import gradio as gr
import os, json, time, subprocess, shlex, traceback
from pathlib import Path
import requests
from openai import OpenAI

# ---- Config ----
COMFY_URL = "http://127.0.0.1:8188"
COMFY_ROOT = Path("/workspace/ComfyUI")
OUTPUT_DIR = COMFY_ROOT / "output"
MODELS_DIR = COMFY_ROOT / "models"
CUSTOM_NODES_DIR = COMFY_ROOT / "custom_nodes"
API_KEY_FILE = Path("/workspace/.openrouter_key")
DEFAULT_MODEL = "deepseek/deepseek-v3.2"

def fetch_openrouter_models():
    """Pull the full model list from OpenRouter. Tool-calling models surface first."""
    try:
        r = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        data = r.json().get("data", [])
        ids = []
        for m in data:
            mid = m.get("id")
            if not mid:
                continue
            params = m.get("supported_parameters") or []
            has_tools = "tools" in params or "tool_choice" in params
            ids.append((not has_tools, mid.lower(), mid))  # tool-capable first, then alpha
        ids.sort()
        return [mid for _, _, mid in ids]
    except Exception as e:
        print(f"[warn] OpenRouter model fetch failed: {e}")
        return [DEFAULT_MODEL, "anthropic/claude-opus-4.7", "anthropic/claude-sonnet-4.6",
                "x-ai/grok-4-fast", "openai/gpt-5", "google/gemini-2.5-pro"]

MODEL_CHOICES = fetch_openrouter_models()
if DEFAULT_MODEL not in MODEL_CHOICES:
    MODEL_CHOICES.insert(0, DEFAULT_MODEL)

SYSTEM_PROMPT = """You are an autonomous ComfyUI operator running on an RTX 5090 (32GB VRAM) Linux box.
ComfyUI is already running at http://127.0.0.1:8188 (its API). You have FULL shell access.

Paths:
  - Models: /workspace/ComfyUI/models/{checkpoints,loras,controlnet,vae,text_encoders,...}
  - Custom nodes: /workspace/ComfyUI/custom_nodes/
  - Outputs: /workspace/ComfyUI/output/
  - ComfyUI venv python: /workspace/ComfyUI/venv/bin/python

Model library is backed by DO Spaces (bucket: imagination-models). Every model
you download gets auto-ingested so the next spawn has it without re-downloading.
Installed custom nodes: ComfyUI-Manager, ComfyUI-AnimateDiff-Evolved, comfyui_controlnet_aux, ComfyUI_IPAdapter_plus, ComfyUI-Custom-Scripts.

RULES OF ENGAGEMENT:
1. Execute — don't describe. When the user asks for an image, CALL queue_workflow. Do not print JSON for them to copy.
2. MODEL SOURCING — manifest-first, always. Before you ever call install_model:
   a. search_library(query=...) — find what's already in our backed-up library
   b. If a match exists: pull_model(filename=...) — fast pull from DO Spaces
   c. list_models('checkpoints') — confirm what's on local disk already
   d. Only if nothing matches, call install_model(url, dest_type). It auto-applies
      HF/Civitai auth and auto-ingests into the manifest on success. Never call
      run_shell to wget a model yourself — that bypasses the backup pipeline.
3. Workflows use ComfyUI's API format: a dict of node_id -> {class_type, inputs}. Inputs that come from other nodes use [node_id, output_index].
3a. EVERY workflow MUST include a terminal output node — typically SaveImage (inputs: images=[vae_decode_id,0], filename_prefix="..."). Without one ComfyUI rejects with "prompt_no_outputs". A minimal SDXL graph needs: CheckpointLoaderSimple → CLIPTextEncode (pos) + CLIPTextEncode (neg) → EmptyLatentImage → KSampler → VAEDecode → SaveImage.
4. After queue_workflow, call wait_for_image(prompt_id) to block until done, then report the output filename.
5. If a workflow fails, read the error from ComfyUI (run_shell: `tail /workspace/comfy.log`), fix, retry.
6. If a node or model is missing, install it (install_custom_node / install_model), then retry. Some custom nodes require restarting ComfyUI — use run_shell to kill + relaunch the tmux session 'comfyui':
     tmux kill-session -t comfyui
     tmux new-session -d -s comfyui 'cd /workspace/ComfyUI && source venv/bin/activate && python main.py --listen 0.0.0.0 --port 8188 2>&1 | tee /workspace/comfy.log'
7. Introspect if unsure: `curl -s http://127.0.0.1:8188/object_info | python3 -c "import sys,json; d=json.load(sys.stdin); print(list(d)[:50])"`.
8. Be terse in chat. Tool output is shown separately — don't re-narrate it.
9. REVIEW LOOP: After generating an image, call review_image on it. If the verdict is 'fail' or 'flag' (rating < 7), read the prompt deltas, revise your workflow accordingly, and regenerate. Iterate up to 3 times. If it still fails, show the user what you have and ask for guidance."""

# Grok-specific: stricter template. Grok struggles with open-ended workflow
# construction, so we hand it an exact skeleton and ask it to substitute.
GROK_SYSTEM_PROMPT = """You generate ComfyUI workflows by CALLING the queue_workflow tool. Always pass this exact structure as the `workflow` argument, substituting the bracketed placeholders:

{
  "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "juggernaut_x_v10_nsfw.safetensors"}},
  "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "[USER PROMPT HERE]", "clip": ["1", 1]}},
  "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, ugly, deformed", "clip": ["1", 1]}},
  "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
  "5": {"class_type": "KSampler", "inputs": {"seed": [RANDOM], "steps": 25, "cfg": 7, "sampler_name": "euler_a", "scheduler": "normal", "denoise": 1, "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
  "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
  "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "[SUBJECT]", "images": ["6", 0]}}
}

Rules:
- Node 2 text = user's creative prompt
- Node 7 filename_prefix = subject name (lowercase, no spaces)
- Seed = random integer 1-9999999
- Call queue_workflow with this dict as the `workflow` argument — do not print JSON to chat
- For NSFW: prepend "score_9, score_8_up, explicit" to node 2 text
- After queue_workflow returns a prompt_id, call wait_for_image(prompt_id), then report the filename briefly

Example: User says "snowman" → node 2 text = "cheerful snowman with carrot nose, winter scene, 4k", node 7 filename_prefix = "snowman"
"""

def system_prompt_for(model: str) -> str:
    """Route to a model-specific system prompt. Grok gets the rigid template."""
    if "grok" in (model or "").lower():
        return GROK_SYSTEM_PROMPT
    return SYSTEM_PROMPT

# ---- Tool implementations ----

def t_queue_workflow(workflow):
    r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": workflow}, timeout=30)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}", "body": r.text[:2000]}
    return r.json()

def t_get_queue_status():
    return requests.get(f"{COMFY_URL}/queue", timeout=10).json()

def t_get_history(n=5):
    all_hist = requests.get(f"{COMFY_URL}/history", timeout=10).json()
    ids = list(all_hist.keys())[-int(n):]
    out = {}
    for pid in ids:
        h = all_hist[pid]
        outs = h.get("outputs", {})
        files = []
        for node_id, node_out in outs.items():
            for img in node_out.get("images", []):
                files.append(img["filename"])
        out[pid] = {"files": files, "status": h.get("status", {}).get("status_str", "unknown")}
    return out

def t_wait_for_image(prompt_id, timeout_s=180):
    deadline = time.time() + int(timeout_s)
    while time.time() < deadline:
        h = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=5).json()
        if prompt_id in h:
            outputs = h[prompt_id].get("outputs", {})
            files = []
            for node_id, node_out in outputs.items():
                for img in node_out.get("images", []):
                    path = OUTPUT_DIR / img.get("subfolder", "") / img["filename"]
                    files.append({
                        "filename": img["filename"],
                        "path": str(path),
                        "view_url": f"{COMFY_URL}/view?filename={img['filename']}&subfolder={img.get('subfolder','')}&type={img.get('type','output')}",
                    })
            if files:
                return {"status": "done", "files": files, "elapsed_s": round(time.time() - (deadline - int(timeout_s)), 1)}
        time.sleep(1.5)
    return {"status": "timeout", "after_s": timeout_s}

def t_list_models(type="checkpoints"):
    d = MODELS_DIR / type
    if not d.exists():
        return {"error": f"{d} not found", "available_types": [p.name for p in MODELS_DIR.iterdir() if p.is_dir()]}
    files = []
    for p in d.iterdir():
        if p.is_file() and not p.name.startswith("put_"):
            files.append({"name": p.name, "size_gb": round(p.stat().st_size / 1e9, 2)})
    return files

_DEST_TYPE_TO_CATEGORY = {
    "checkpoints": "checkpoint", "loras": "lora", "vae": "vae",
    "controlnet": "controlnet", "embeddings": "embedding",
    "upscalers": "upscaler", "text_encoders": "text_encoder",
}

_REPO_ROOT = Path(__file__).resolve().parent

def _install_model_preflight(filename):
    """If filename is in the manifest, return the row. Else None."""
    conn = _get_manifest_conn()
    if not conn:
        return None
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from library.manifest import get_by_filename
    row = get_by_filename(conn, filename)
    conn.close()
    return row

def t_install_model(url, dest_type, filename=None):
    """Install a model. Manifest-first: if filename is already in our library,
    pull from DO Spaces (fast). Otherwise download from `url` (with HF/Civitai
    auth), atomic-rename on success, and auto-ingest so the next spawn has it
    in the library without re-downloading from source."""
    dest_dir = MODELS_DIR / dest_type
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not filename:
        filename = url.rsplit("/", 1)[-1].split("?")[0]
    dest = dest_dir / filename

    if dest.exists() and dest.stat().st_size > 0:
        return {"status": "already_on_disk", "path": str(dest)}

    # Manifest preflight — route to Spaces pull if we already have this model
    row = _install_model_preflight(filename)
    if row and row.get("spaces_key"):
        return t_pull_model(filename=filename)

    # Not in manifest — download from source with appropriate auth
    auth_flags = ""
    hf_tok = os.environ.get("HF_TOKEN", "")
    civ_tok = os.environ.get("CIVITAI_API_TOKEN", "")
    if "huggingface.co" in url and hf_tok:
        auth_flags = "--header=" + shlex.quote("Authorization: Bearer " + hf_tok)
    elif "civitai.com" in url and civ_tok:
        sep = "&" if "?" in url else "?"
        url = url + sep + "token=" + civ_tok

    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    category = _DEST_TYPE_TO_CATEGORY.get(dest_type, "other")
    tmux_name = f"dl_{int(time.time())}"
    log_path = f"/tmp/{tmux_name}.log"
    py = "/workspace/ComfyUI/venv/bin/python"

    # Download → atomic rename → ingest (upload to Spaces + add manifest row)
    # On any failure, clean up the .part file so we don't leave a stub.
    cmd = (
        f"set -e; "
        f"wget --progress=dot:giga {auth_flags} -O {shlex.quote(str(tmp_dest))} {shlex.quote(url)} "
        f"&& mv {shlex.quote(str(tmp_dest))} {shlex.quote(str(dest))} "
        f"&& cd {shlex.quote(str(_REPO_ROOT))} "
        f"&& {py} ingest.py local {shlex.quote(str(dest))} --category {category} "
        f"|| {{ echo INSTALL_FAILED; rm -f {shlex.quote(str(tmp_dest))}; exit 1; }}"
    )
    full_cmd = f"({cmd}) 2>&1 | tee {log_path}"
    subprocess.Popen(["tmux", "new-session", "-d", "-s", tmux_name, full_cmd])
    return {
        "status": "download_started",
        "tmux_session": tmux_name,
        "dest": str(dest),
        "auto_ingest": True,
        "check_progress": f"run_shell: tail -20 {log_path}",
        "done_when": "log ends with a success line from ingest.py (look for 'Manifest pushed to Spaces' or 'Done:')",
    }

def t_pull_model(filename):
    """Pull a model from DO Spaces by filename. Model must be in the manifest.
    Returns quickly — the s5cmd transfer runs in a tmux session."""
    row = _install_model_preflight(filename)
    if not row:
        return {"error": f"'{filename}' not in manifest. Use install_model(url, dest_type) to fetch from source."}
    spaces_key = row["spaces_key"]
    # spaces_key like "models/checkpoints/foo.safetensors" — strip "models/" for local path
    rel = spaces_key[len("models/"):] if spaces_key.startswith("models/") else spaces_key
    dest = MODELS_DIR / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return {"status": "already_on_disk", "path": str(dest),
                "display_name": row["display_name"], "trigger_words": row["trigger_words"]}
    bucket = os.environ.get("COMFY_S3_BUCKET", "imagination-models")
    endpoint = os.environ.get("COMFY_S3_ENDPOINT", "https://sfo3.digitaloceanspaces.com")
    access = os.environ.get("COMFY_S3_ACCESS_KEY", "")
    secret = os.environ.get("COMFY_S3_SECRET_KEY", "")
    if not (access and secret):
        return {"error": "S3 credentials not available in env."}
    tmux_name = f"dl_{int(time.time())}"
    log_path = f"/tmp/{tmux_name}.log"
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    cmd = (
        f"AWS_ACCESS_KEY_ID={shlex.quote(access)} AWS_SECRET_ACCESS_KEY={shlex.quote(secret)} "
        f"s5cmd --endpoint-url {shlex.quote(endpoint)} cp "
        f"{shlex.quote(f's3://{bucket}/{spaces_key}')} {shlex.quote(str(tmp_dest))} "
        f"&& mv {shlex.quote(str(tmp_dest))} {shlex.quote(str(dest))} "
        f"|| {{ echo PULL_FAILED; rm -f {shlex.quote(str(tmp_dest))}; exit 1; }}"
    )
    full_cmd = f"({cmd}) 2>&1 | tee {log_path}"
    subprocess.Popen(["tmux", "new-session", "-d", "-s", tmux_name, full_cmd])
    return {
        "status": "pull_started",
        "tmux_session": tmux_name,
        "dest": str(dest),
        "display_name": row["display_name"],
        "trigger_words": row["trigger_words"],
        "check_progress": f"run_shell: tail -5 {log_path}",
    }

def t_list_custom_nodes():
    return [p.name for p in CUSTOM_NODES_DIR.iterdir() if p.is_dir()]

def t_install_custom_node(repo_url):
    name = repo_url.rstrip("/").rsplit("/", 1)[-1].replace(".git", "")
    dest = CUSTOM_NODES_DIR / name
    if dest.exists():
        return {"status": "already_installed", "name": name}
    r = subprocess.run(["git", "clone", repo_url, str(dest)], capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        return {"status": "clone_failed", "stderr": r.stderr[-1000:]}
    req = dest / "requirements.txt"
    pip_tail = ""
    if req.exists():
        p = subprocess.run(
            ["/workspace/ComfyUI/venv/bin/pip", "install", "-r", str(req)],
            capture_output=True, text=True, timeout=600,
        )
        pip_tail = (p.stdout + p.stderr)[-800:]
    return {
        "status": "installed",
        "name": name,
        "pip_tail": pip_tail,
        "note": "Restart ComfyUI to load the new node (see rule #6).",
    }

def t_gpu_status():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    )
    return r.stdout.strip()

def t_run_shell(cmd, timeout_s=60):
    try:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=int(timeout_s))
        return {
            "returncode": r.returncode,
            "stdout": r.stdout[-3000:],
            "stderr": r.stderr[-1500:],
        }
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout_s}s"}

def t_object_info():
    """Fetch ComfyUI's node catalog (abbreviated) so you know what node class_types exist."""
    r = requests.get(f"{COMFY_URL}/object_info", timeout=30).json()
    # Just names (full catalog is huge)
    return {"count": len(r), "names": sorted(r.keys())}

# ---- Library tools (manifest-backed model search) ----

MANIFEST_DB = Path("/workspace/library/manifest.db")

def _get_manifest_conn():
    """Open the manifest DB. Returns conn or None if not available."""
    if not MANIFEST_DB.exists():
        return None
    import sqlite3
    conn = sqlite3.connect(str(MANIFEST_DB))
    conn.row_factory = sqlite3.Row
    return conn

def t_search_library(query="", base_model="", category=""):
    """Fuzzy search the model library manifest."""
    conn = _get_manifest_conn()
    if not conn:
        return {"error": "Manifest not available. Run model sync or ingest first."}
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from library.manifest import search
    results = search(conn, query, base_model=base_model or None, category=category or None)
    conn.close()
    # Return concise results for LLM context
    return [
        {
            "filename": r["filename"],
            "display_name": r["display_name"],
            "category": r["category"],
            "base_model": r["base_model"],
            "trigger_words": r["trigger_words"],
            "weight_range": r["weight_range"],
        }
        for r in results
    ]

def t_get_lora_details(filename=""):
    """Get full metadata for a model by filename."""
    conn = _get_manifest_conn()
    if not conn:
        return {"error": "Manifest not available."}
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from library.manifest import get_by_filename
    result = get_by_filename(conn, filename)
    conn.close()
    if not result:
        return {"error": f"'{filename}' not found in manifest."}
    return result

def t_list_checkpoints(base_model=""):
    """List available checkpoints, optionally filtered by base model."""
    conn = _get_manifest_conn()
    if not conn:
        return {"error": "Manifest not available."}
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from library.manifest import list_checkpoints
    results = list_checkpoints(conn, base_model=base_model or None)
    conn.close()
    return [
        {
            "filename": r["filename"],
            "display_name": r["display_name"],
            "base_model": r["base_model"],
            "notes": r["notes"],
        }
        for r in results
    ]


def _get_openrouter_key():
    """Read the persisted OpenRouter API key."""
    if API_KEY_FILE.exists():
        return API_KEY_FILE.read_text().strip()
    return None

VISION_MODEL = "x-ai/grok-4.1-fast"

def t_review_image(image_path, focus="overall quality, composition, and style"):
    """Send a generated image to Grok vision for quality review.

    Returns a structured verdict: pass/flag/fail with a numeric rating,
    what works, what doesn't, and specific revision notes for regeneration.
    """
    import base64
    api_key = _get_openrouter_key()
    if not api_key:
        return {"error": "No OpenRouter API key set. Enter one in the UI."}

    # Resolve the path — check output dir, workspace, and absolute
    p = Path(image_path)
    if not p.is_absolute():
        candidates = [
            OUTPUT_DIR / image_path,
            COMFY_ROOT / image_path,
            Path("/workspace") / image_path,
        ]
        for c in candidates:
            if c.exists():
                p = c
                break
    if not p.exists():
        return {"error": f"File not found: {image_path}"}
    if p.stat().st_size > 10_000_000:
        return {"error": f"File too large ({p.stat().st_size / 1e6:.1f} MB). Max 10 MB."}

    img_b64 = base64.b64encode(p.read_bytes()).decode()
    ext = p.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "webp": "image/webp", "gif": "image/gif"}.get(ext, "image/png")

    system_text = (
        "You are a film director reviewing a generated image.\n"
        f"Focus: {focus}\n\n"
        "Respond with ONLY a JSON object (no markdown, no extra text):\n"
        '{"rating": 1-10, "verdict": "pass|flag|fail", '
        '"strengths": "what works well", '
        '"weaknesses": "what needs improvement", '
        '"positive_prompt_delta": "what to add or keep in the prompt", '
        '"negative_prompt_delta": "what to remove or avoid in the prompt", '
        '"recommendation": "approve|iterate|rethink"}\n\n'
        "Rating guide: 7+ = pass, 4-6 = flag (fixable issues), 1-3 = fail (start over)."
    )

    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text", "text": system_text},
            ]}],
        )
        review_text = resp.choices[0].message.content or ""

        # Parse structured verdict
        import re as _re
        try:
            verdict = json.loads(review_text)
        except json.JSONDecodeError:
            m = _re.search(r'\{.*\}', review_text, _re.DOTALL)
            if m:
                try:
                    verdict = json.loads(m.group(0))
                except json.JSONDecodeError:
                    verdict = None
            else:
                verdict = None

        if not verdict or not isinstance(verdict, dict):
            verdict = {"rating": 5, "verdict": "flag", "raw_review": review_text,
                        "recommendation": "iterate"}

        verdict.setdefault("rating", 5)
        verdict.setdefault("verdict", "flag" if verdict["rating"] < 7 else "pass")
        verdict["image_path"] = image_path
        verdict["focus"] = focus
        return verdict

    except Exception as e:
        return {"error": f"Vision review failed: {e}"}


TOOL_FNS = {
    "queue_workflow": t_queue_workflow,
    "get_queue_status": t_get_queue_status,
    "get_history": t_get_history,
    "wait_for_image": t_wait_for_image,
    "list_models": t_list_models,
    "install_model": t_install_model,
    "pull_model": t_pull_model,
    "list_custom_nodes": t_list_custom_nodes,
    "install_custom_node": t_install_custom_node,
    "gpu_status": t_gpu_status,
    "run_shell": t_run_shell,
    "object_info": t_object_info,
    "search_library": t_search_library,
    "get_lora_details": t_get_lora_details,
    "list_checkpoints": t_list_checkpoints,
    "review_image": t_review_image,
}

# ---- Tool schemas (OpenAI function-calling) ----
TOOLS = [
    {"type": "function", "function": {
        "name": "queue_workflow",
        "description": "Submit a ComfyUI API-format workflow for execution. Returns prompt_id.",
        "parameters": {"type": "object", "properties": {
            "workflow": {"type": "object", "description": "Dict of node_id -> {class_type, inputs}. Inputs referencing other nodes are [node_id_string, output_index_int]."}
        }, "required": ["workflow"]}
    }},
    {"type": "function", "function": {
        "name": "get_queue_status",
        "description": "Return ComfyUI's current queue (running + pending).",
        "parameters": {"type": "object", "properties": {}}
    }},
    {"type": "function", "function": {
        "name": "get_history",
        "description": "Return the last N completed prompts and their output filenames.",
        "parameters": {"type": "object", "properties": {"n": {"type": "integer", "default": 5}}}
    }},
    {"type": "function", "function": {
        "name": "wait_for_image",
        "description": "Poll /history/{prompt_id} until outputs appear or timeout. Returns file paths.",
        "parameters": {"type": "object", "properties": {
            "prompt_id": {"type": "string"},
            "timeout_s": {"type": "integer", "default": 180}
        }, "required": ["prompt_id"]}
    }},
    {"type": "function", "function": {
        "name": "list_models",
        "description": "List files in /workspace/ComfyUI/models/<type>/. type defaults to 'checkpoints'.",
        "parameters": {"type": "object", "properties": {"type": {"type": "string", "default": "checkpoints"}}}
    }},
    {"type": "function", "function": {
        "name": "install_model",
        "description": "Download a model from an external URL (HuggingFace / CivitAI / etc.). Manifest-first: if the filename is already in our library manifest, this routes to pull_model (fast Spaces pull) instead. On fresh download, auth is auto-applied (HF_TOKEN / CIVITAI_API_TOKEN) and the file is auto-ingested into the manifest + backed up to Spaces, so next spawn has it available. Returns immediately; check progress via run_shell.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"},
            "dest_type": {"type": "string", "description": "e.g. checkpoints, loras, controlnet, vae, text_encoders"},
            "filename": {"type": "string", "description": "Override filename (default: infer from URL)"}
        }, "required": ["url", "dest_type"]}
    }},
    {"type": "function", "function": {
        "name": "pull_model",
        "description": "Pull a model from our DO Spaces backup by filename. The model MUST already be in the manifest (use search_library first). This is the fast path — prefer it over install_model when the library already has what you need.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "Exact filename as stored in the manifest"}
        }, "required": ["filename"]}
    }},
    {"type": "function", "function": {
        "name": "list_custom_nodes",
        "description": "List installed custom nodes.",
        "parameters": {"type": "object", "properties": {}}
    }},
    {"type": "function", "function": {
        "name": "install_custom_node",
        "description": "git clone a custom node into /workspace/ComfyUI/custom_nodes/ and pip install its requirements. Requires ComfyUI restart to activate.",
        "parameters": {"type": "object", "properties": {
            "repo_url": {"type": "string"}
        }, "required": ["repo_url"]}
    }},
    {"type": "function", "function": {
        "name": "gpu_status",
        "description": "One-line nvidia-smi summary: name, mem used/total, util%, temp.",
        "parameters": {"type": "object", "properties": {}}
    }},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "Run an arbitrary bash command. stdout/stderr truncated. Use for anything else.",
        "parameters": {"type": "object", "properties": {
            "cmd": {"type": "string"},
            "timeout_s": {"type": "integer", "default": 60}
        }, "required": ["cmd"]}
    }},
    {"type": "function", "function": {
        "name": "object_info",
        "description": "Fetch all registered ComfyUI node class_type names (no schemas — just names).",
        "parameters": {"type": "object", "properties": {}}
    }},
    # Library tools (manifest-backed)
    {"type": "function", "function": {
        "name": "search_library",
        "description": "Search the model library for LoRAs, checkpoints, embeddings, etc. Returns matching models with trigger words, weight ranges, and base model compatibility. Use this to find the right model for a generation task.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search term (name, style, concept, etc.)"},
            "base_model": {"type": "string", "description": "Filter by base model: SDXL, Flux, SD1.5, Pony. Omit for all."},
            "category": {"type": "string", "description": "Filter by type: lora, checkpoint, vae, controlnet, embedding. Omit for all."},
        }}
    }},
    {"type": "function", "function": {
        "name": "get_lora_details",
        "description": "Get full metadata for a specific model by filename — trigger words, recommended weight range, base model, source, notes.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "Exact filename (e.g. 'anime_style.safetensors')"},
        }, "required": ["filename"]}
    }},
    {"type": "function", "function": {
        "name": "list_checkpoints",
        "description": "List all available checkpoint models, optionally filtered by base model compatibility.",
        "parameters": {"type": "object", "properties": {
            "base_model": {"type": "string", "description": "Filter: SDXL, Flux, SD1.5, Pony. Omit for all."},
        }}
    }},
    {"type": "function", "function": {
        "name": "review_image",
        "description": "Send a generated image to Grok vision for quality review. Returns a structured verdict (pass/flag/fail) with a 1-10 rating, strengths, weaknesses, and specific prompt revision notes. Use after generating an image to check quality. If verdict is 'fail' or 'flag', use the prompt deltas to revise your workflow and regenerate. Iterate up to 3 times.",
        "parameters": {"type": "object", "properties": {
            "image_path": {"type": "string", "description": "Path to the image file (can be relative to output dir, e.g. 'ComfyUI_00042_.png')"},
            "focus": {"type": "string", "description": "What to focus on in the review (e.g. 'character consistency', 'lighting quality', 'overall composition'). Default: general quality."},
        }, "required": ["image_path"]}
    }},
]

def dispatch(name, args):
    fn = TOOL_FNS.get(name)
    if not fn:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": f"bad args: {e}", "got": args}
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__, "tb": traceback.format_exc()[-1000:]}

# ---- Chat loop ----
MAX_STEPS = 50

def chat(user_msg, display_hist, api_hist, api_key, model):
    if not user_msg.strip():
        yield "", display_hist, api_hist
        return
    if not api_key:
        display_hist = display_hist + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "⚠️ Enter your OpenRouter API key above first."}
        ]
        yield "", display_hist, api_hist
        return

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Seed api_hist with system prompt if empty.
    # NOTE: the prompt is chosen per-model at seed time. To switch prompts
    # when you change models mid-session, hit Clear first.
    if not api_hist:
        api_hist = [{"role": "system", "content": system_prompt_for(model)}]

    api_hist = api_hist + [{"role": "user", "content": user_msg}]
    display_hist = display_hist + [{"role": "user", "content": user_msg}]
    yield "", display_hist, api_hist

    for step in range(MAX_STEPS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=api_hist,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=4096,
            )
        except Exception as e:
            display_hist = display_hist + [{"role": "assistant", "content": f"❌ API error: {e}"}]
            yield "", display_hist, api_hist
            return

        # OpenRouter can return a 200 with no `.choices` (e.g. upstream rate-limit,
        # content filter, invalid model slug). Surface the underlying error instead
        # of crashing with NoneType[0].
        if not getattr(resp, "choices", None):
            err = getattr(resp, "error", None) or getattr(resp, "model_dump", lambda: {})()
            display_hist = display_hist + [{
                "role": "assistant",
                "content": f"❌ OpenRouter returned no choices for `{model}`:\n```\n{err}\n```",
            }]
            yield "", display_hist, api_hist
            return

        m = resp.choices[0].message
        # Append the assistant message (with tool_calls if any) to API history
        assistant_entry = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            assistant_entry["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in m.tool_calls
            ]
        api_hist = api_hist + [assistant_entry]

        # Show any text the assistant emitted before/alongside the tool calls
        if m.content:
            display_hist = display_hist + [{"role": "assistant", "content": m.content}]
            yield "", display_hist, api_hist

        if not m.tool_calls:
            return  # final answer already displayed

        # Execute tools
        for tc in m.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            # Show a "calling..." message
            pretty_args = json.dumps(args, indent=2)[:400]
            display_hist = display_hist + [{
                "role": "assistant",
                "content": f"🔧 **{name}** `{pretty_args}`"
            }]
            yield "", display_hist, api_hist

            result = dispatch(name, args)
            result_str = json.dumps(result, default=str)[:8000]

            api_hist = api_hist + [{
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            }]

            # Condensed display
            preview = json.dumps(result, default=str, indent=2)
            if len(preview) > 800:
                preview = preview[:800] + "\n…(truncated)"
            display_hist = display_hist + [{
                "role": "assistant",
                "content": f"↳ `{name}` result:\n```json\n{preview}\n```"
            }]
            yield "", display_hist, api_hist

    display_hist = display_hist + [{"role": "assistant", "content": "⚠️ Max tool-call steps reached."}]
    yield "", display_hist, api_hist


# ---- Gallery helpers ----
def refresh_gallery():
    if not OUTPUT_DIR.exists():
        return []
    files = sorted(
        [p for p in OUTPUT_DIR.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )[:60]
    return [(str(p), p.name) for p in files]

def save_key(k):
    if k:
        try:
            API_KEY_FILE.write_text(k.strip())
        except Exception:
            pass
    return gr.update()

FORAGE_DROPLET = "http://jobscout.tail2b8e3e.ts.net:5001"

def _get_ts_hostname():
    """Read Tailscale hostname from env or persisted file."""
    h = os.environ.get("TS_HOSTNAME", "")
    if not h:
        try:
            h = Path("/workspace/.ts_hostname").read_text().strip()
        except FileNotFoundError:
            pass
    return h

def sync_to_n8razer():
    """Call the Forage droplet to sync ComfyUI outputs to N8Razer's 10TB drive."""
    ts_hostname = _get_ts_hostname()
    if not ts_hostname:
        return gr.update(value="No Tailscale hostname set (TS_HOSTNAME)", visible=True)
    try:
        r = requests.post(
            f"{FORAGE_DROPLET}/imagination/gpu/sync",
            json={"ts_hostname": ts_hostname},
            timeout=300,
        )
        data = r.json()
        if not r.ok:
            return gr.update(value=f"Sync error: {data.get('error', 'unknown')}", visible=True)
        msg = f"Synced {data['synced']} files, skipped {data['skipped']} existing"
        if data.get("errors"):
            msg += f", {len(data['errors'])} errors"
        msg += f" → `{data.get('dest', '?')}`"
        return gr.update(value=msg, visible=True)
    except Exception as e:
        return gr.update(value=f"Sync failed: {e}", visible=True)

saved_key = API_KEY_FILE.read_text().strip() if API_KEY_FILE.exists() else ""

# ---- UI ----
CSS = """
footer {display:none !important}
.gradio-container {max-width: 100% !important; padding: 8px !important;}
.gradio-container > .main, .gradio-container .contain {max-width: 100% !important; padding: 0 !important;}
#comfyframe {min-height: 900px;}
#comfyframe iframe {width: 100%; height: 900px; border: 1px solid #444; border-radius: 8px;}
.chatbot {height: 560px !important;}
"""

with gr.Blocks(title="ComfyUI Studio", fill_height=True) as demo:
    gr.Markdown("## 🎨 ComfyUI Studio — chat + canvas")
    api_state = gr.State([])

    with gr.Row():
        # Left: chat (20%)
        with gr.Column(scale=1, min_width=320):
            with gr.Row():
                model_pick = gr.Dropdown(
                    choices=MODEL_CHOICES, value=DEFAULT_MODEL, label="Model",
                    filterable=True, allow_custom_value=True, scale=4,
                )
                edit_key_btn = gr.Button("🔑", scale=0, min_width=44,
                                         visible=bool(saved_key))
            api_key = gr.Textbox(
                value=saved_key, label="OpenRouter Key", type="password",
                visible=not bool(saved_key),
            )
            chatbot = gr.Chatbot(elem_classes=["chatbot"], show_label=False,
                                 avatar_images=(None, "https://openrouter.ai/favicon.ico"))
            msg = gr.Textbox(placeholder="e.g., Generate a cyberpunk portrait at 1024x1024 with Juggernaut",
                             lines=2, show_label=False)
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary", scale=3)
                clear_btn = gr.Button("Clear", scale=1)
            gr.HTML(
                '<a href="http://localhost:7681" target="_blank" rel="noopener" '
                'style="display:block;text-align:center;padding:8px 12px;margin-top:4px;'
                'background:#b33;color:#fff;border-radius:6px;text-decoration:none;'
                'font-weight:600;">🛟 Rescue terminal (Claude Code)</a>'
            )
            with gr.Accordion("Recent outputs", open=True):
                gallery = gr.Gallery(value=refresh_gallery(), columns=4, height=480,
                                     show_label=False, allow_preview=True)
                with gr.Row():
                    refresh_btn = gr.Button("Refresh", size="sm", scale=1)
                    sync_btn = gr.Button("💾 Save to N8Razer", size="sm", scale=1,
                                          variant="secondary")
                sync_status = gr.Markdown("", visible=False)
        # Right: ComfyUI iframe (80%)
        with gr.Column(scale=4, elem_id="comfyframe"):
            _comfy_host = _get_ts_hostname() or "localhost"
            _comfy_url = f"http://{_comfy_host}.tail2b8e3e.ts.net:8188" if _comfy_host and _comfy_host != "localhost" else "http://localhost:8188"
            gr.HTML(f'<iframe src="{_comfy_url}" allow="clipboard-write"></iframe>')

    # Wiring
    send_btn.click(chat, [msg, chatbot, api_state, api_key, model_pick],
                   [msg, chatbot, api_state]).then(refresh_gallery, None, gallery)
    msg.submit(chat, [msg, chatbot, api_state, api_key, model_pick],
               [msg, chatbot, api_state]).then(refresh_gallery, None, gallery)
    clear_btn.click(lambda: ([], []), None, [chatbot, api_state])
    refresh_btn.click(refresh_gallery, None, gallery)
    sync_btn.click(sync_to_n8razer, None, sync_status)
    api_key.change(save_key, api_key, None)
    edit_key_btn.click(lambda: gr.update(visible=True), None, api_key)

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=3000,
        share=False,
        allowed_paths=[str(OUTPUT_DIR)],
        css=CSS,
        theme=gr.themes.Soft(),
    )

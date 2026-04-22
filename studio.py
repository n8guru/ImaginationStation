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
9. REVIEW LOOP: After generating an image, call review_image on it. If the verdict is 'fail' or 'flag' (rating < 7), read the prompt deltas, revise your workflow accordingly, and regenerate. Iterate up to 3 times. If it still fails, show the user what you have and ask for guidance.
10. REFERENCE IMAGES: When the user's message starts with [Reference images: ...], it contains the FULL workflow that produced those images — prompt, negative prompt, checkpoint, and LoRAs. This is YOUR starting point. You already have everything you need. Do NOT ask the user to repeat any of this. Build your workflow directly from the provided metadata, applying whatever changes the user requests. If the user says "same but change X", use the exact same prompt/checkpoint/loras and only modify X."""

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

    # Extract workflow metadata from PNG so the reviewer knows what was intended
    meta = _extract_png_meta(str(p))
    intent_context = ""
    if meta:
        if meta["positive"]:
            intent_context += f"\nINTENDED PROMPT (what the image should depict):\n{meta['positive']}\n"
        if meta["negative"]:
            intent_context += f"\nNEGATIVE PROMPT (what should NOT appear):\n{meta['negative']}\n"
        if meta["checkpoint"]:
            intent_context += f"\nCheckpoint: {meta['checkpoint']}\n"
        if meta["loras"]:
            intent_context += f"LoRAs: {', '.join(meta['loras'])}\n"

    system_text = (
        "You are reviewing a generated image against its intended prompt.\n"
        f"Focus: {focus}\n"
        f"{intent_context}\n"
        "REVIEW PRIORITIES (in order):\n"
        "1. FACTUAL ACCURACY: Correct number of people? Correct body positions/poses as described?\n"
        "   Count heads, limbs, bodies. Flag extra/missing people immediately.\n"
        "2. ANATOMICAL CORRECTNESS: Right number of hands, fingers, limbs? Natural proportions?\n"
        "3. PROMPT ADHERENCE: Does the scene match what the prompt describes?\n"
        "4. COMPOSITION & QUALITY: Lighting, framing, style.\n\n"
        "Respond with ONLY a JSON object (no markdown, no extra text):\n"
        '{"rating": 1-10, "verdict": "pass|flag|fail", '
        '"people_count": "expected N, found N", '
        '"position_correct": true/false, '
        '"anatomy_issues": "list any problems or empty string", '
        '"strengths": "what works well", '
        '"weaknesses": "what needs improvement", '
        '"positive_prompt_delta": "what to add or keep in the prompt", '
        '"negative_prompt_delta": "what to remove or avoid in the prompt", '
        '"recommendation": "approve|iterate|rethink"}\n\n'
        "Rating guide: 7+ = pass, 4-6 = flag (fixable issues), 1-3 = fail (start over).\n"
        "Wrong number of people or badly wrong positions = automatic fail (rating 1-3)."
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


WORKFLOWS_DIR = Path("/workspace/workflows")

def t_list_workflows():
    """List available reference workflow JSONs (Daxamur Wan 2.2 etc)."""
    if not WORKFLOWS_DIR.exists():
        # Pull from Spaces on first call
        bucket = os.environ.get("COMFY_S3_BUCKET", "imagination-models")
        endpoint = os.environ.get("COMFY_S3_ENDPOINT", "https://sfo3.digitaloceanspaces.com")
        ak = os.environ.get("COMFY_S3_ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID", "")
        sk = os.environ.get("COMFY_S3_SECRET_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        if ak and sk:
            WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)
            cmd = f's5cmd --endpoint-url {endpoint} sync "s3://{bucket}/workflows/*" {WORKFLOWS_DIR}/'
            subprocess.run(cmd, shell=True, timeout=30)
    if not WORKFLOWS_DIR.exists():
        return {"error": "No workflows directory. S3 creds may be missing."}
    result = []
    for p in sorted(WORKFLOWS_DIR.rglob("*.json")):
        result.append({"name": p.name, "path": str(p), "size_kb": round(p.stat().st_size / 1024)})
    return result

def t_load_workflow(path):
    """Load a workflow JSON and return its structure. Works with both ComfyUI
    API format (node_id→node dict) and UI format (nodes+links arrays).
    Returns a summary of models, prompts, settings, and node types used."""
    p = Path(path)
    if not p.exists():
        p = WORKFLOWS_DIR / path
    if not p.exists():
        return {"error": f"Not found: {path}"}
    data = json.loads(p.read_text())

    summary = {}

    # Detect format
    if "nodes" in data and "links" in data:
        # UI format — extract useful info from nodes array
        nodes = data.get("nodes", [])
        summary["format"] = "ui"
        summary["total_nodes"] = len(nodes)

        # Catalog node types and their widget values
        node_types = {}
        models_used = []
        prompts = []
        samplers = []

        for n in nodes:
            ntype = n.get("type", "unknown")
            node_types[ntype] = node_types.get(ntype, 0) + 1
            widgets = n.get("widgets_values", [])

            # Extract model references
            if "Loader" in ntype or "Load" in ntype:
                for w in widgets:
                    if isinstance(w, str) and w.endswith((".safetensors", ".gguf", ".ckpt")):
                        models_used.append({"node_type": ntype, "file": w})

            # Extract text prompts
            if "CLIPText" in ntype or "TextEncode" in ntype or "Text" in ntype:
                for w in widgets:
                    if isinstance(w, str) and len(w) > 20:
                        prompts.append({"node_type": ntype, "node_id": n.get("id"), "text": w[:300]})

            # Extract sampler settings
            if "KSampler" in ntype or "Sampler" in ntype:
                sampler_info = {"node_type": ntype}
                for w in widgets:
                    if isinstance(w, int) and 1 <= w <= 200:
                        sampler_info.setdefault("steps_or_seed", []).append(w)
                    elif isinstance(w, (int, float)) and 0 < w < 30:
                        sampler_info.setdefault("cfg_or_denoise", []).append(w)
                    elif isinstance(w, str) and w in ("euler", "euler_a", "dpmpp_2m", "uni_pc", "ddim", "lms"):
                        sampler_info["sampler_name"] = w
                    elif isinstance(w, str) and w in ("normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta"):
                        sampler_info["scheduler"] = w
                samplers.append(sampler_info)

        summary["node_types"] = dict(sorted(node_types.items(), key=lambda x: -x[1]))
        summary["models_referenced"] = models_used
        if prompts:
            summary["prompts"] = prompts
        if samplers:
            summary["samplers"] = samplers
        summary["note"] = (
            "This is a UI-format workflow (not directly usable with queue_workflow). "
            "Use it as a REFERENCE to understand the pipeline structure, then build "
            "a minimal API-format workflow with the same models and settings. "
            "A basic Wan 2.2 T2V pipeline needs ~10 nodes: 2x LoadDiffusionModel "
            "(high+low noise), LoadCLIP (umt5), LoadVAE, CLIPTextEncode (prompt), "
            "EmptyWanLatentVideo, KSampler (high noise), KSampler (low noise), "
            "VAEDecode, SaveAnimatedWEBP/SaveVideo."
        )

    else:
        # API format — direct summary
        summary["format"] = "api"
        summary["total_nodes"] = len(data) if isinstance(data, dict) else "?"
        if isinstance(data, dict):
            for nid, node in data.items():
                ct = node.get("class_type", "")
                inp = node.get("inputs", {})
                if ct == "CLIPTextEncode" and inp.get("text"):
                    summary.setdefault("prompts", []).append({"node": nid, "text": inp["text"][:200]})
                elif "Loader" in ct and inp.get("ckpt_name"):
                    summary.setdefault("models", []).append(inp["ckpt_name"])
                elif ct == "LoraLoader" and inp.get("lora_name"):
                    summary.setdefault("loras", []).append(inp["lora_name"])
                elif "EmptyLatentImage" in ct or "LatentImage" in ct:
                    if inp.get("width"):
                        summary["resolution"] = f"{inp.get('width')}x{inp.get('height')}"
                elif "KSampler" in ct:
                    summary["sampler"] = {k: inp[k] for k in ["steps", "cfg", "sampler_name", "scheduler", "denoise"] if k in inp}
        summary["full_workflow"] = data

    return summary


def t_save_workflow(name, workflow, notes=""):
    """Save a working workflow as a named template for reuse. The workflow
    argument should be the exact node dict you'd pass to queue_workflow.
    Saved to /workspace/workflows/saved/ and backed up to DO Spaces."""
    WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)
    saved_dir = WORKFLOWS_DIR / "saved"
    saved_dir.mkdir(exist_ok=True)

    # Sanitize name
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in name).strip().replace(" ", "_")
    if not safe_name:
        return {"error": "Invalid name"}
    filename = f"{safe_name}.json"
    path = saved_dir / filename

    # Save with metadata wrapper
    data = {
        "name": name,
        "notes": notes,
        "saved_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "workflow": workflow,
    }
    path.write_text(json.dumps(data, indent=2))

    # Backup to DO Spaces
    bucket = os.environ.get("COMFY_S3_BUCKET", "imagination-models")
    endpoint = os.environ.get("COMFY_S3_ENDPOINT", "https://sfo3.digitaloceanspaces.com")
    ak = os.environ.get("COMFY_S3_ACCESS_KEY", "")
    sk = os.environ.get("COMFY_S3_SECRET_KEY", "")
    if ak and sk:
        try:
            import subprocess
            spaces_key = f"workflows/saved/{filename}"
            subprocess.run([
                "s5cmd", "--endpoint-url", endpoint,
                "cp", str(path), f"s3://{bucket}/{spaces_key}"
            ], timeout=15, capture_output=True)
        except Exception:
            pass  # local save succeeded, S3 backup is best-effort

    # Also copy into ComfyUI user workflows so it appears in the sidebar
    comfy_wf_dir = COMFY_ROOT / "user" / "default" / "workflows"
    comfy_wf_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(str(path), str(comfy_wf_dir / filename))

    return {"status": "saved", "name": name, "path": str(path), "filename": filename}

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
    "list_workflows": t_list_workflows,
    "load_workflow": t_load_workflow,
    "save_workflow": t_save_workflow,
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
    {"type": "function", "function": {
        "name": "list_workflows",
        "description": "List available reference workflow JSONs (Daxamur Wan 2.2 T2V/I2V/FLF2V etc). These are production-tested ComfyUI workflows you can study and adapt. Pull from Spaces on first call.",
        "parameters": {"type": "object", "properties": {}}
    }},
    {"type": "function", "function": {
        "name": "load_workflow",
        "description": "Load a reference workflow JSON and return its structure — prompts, checkpoint, LoRAs, sampler settings, resolution, and the full node graph. Use this to understand how a proven workflow is built before adapting it for your own generation. Pass the path from list_workflows.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path to the workflow JSON file"},
        }, "required": ["path"]}
    }},
    {"type": "function", "function": {
        "name": "save_workflow",
        "description": "Save a working workflow as a named template for reuse. After you've built and tested a workflow that produces good results, save it so you (or the next session) can load it instantly without rebuilding. The workflow is backed up to DO Spaces so it persists across GPU instances.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Human-readable name (e.g. 'SDXL BigLove portrait', 'Wan T2V NSFW missionary')"},
            "workflow": {"type": "object", "description": "The exact node dict — same format you pass to queue_workflow"},
            "notes": {"type": "string", "description": "Optional notes: what models/LoRAs it uses, recommended prompts, settings tips"},
        }, "required": ["name", "workflow"]}
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
MAX_STEPS = 120

def chat(user_msg, display_hist, api_hist, api_key, model, selected_files=""):
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

    # Append selected image references with workflow metadata
    ref_block = ""
    if selected_files and selected_files.strip():
        refs = [f.strip() for f in selected_files.strip().split("\n") if f.strip()]
        if refs:
            ref_parts = []
            for fname in refs:
                fpath = OUTPUT_DIR / fname
                meta = _extract_png_meta(str(fpath)) if fpath.exists() else None
                if meta:
                    parts = [f"  file: {fname}"]
                    if meta["positive"]:
                        parts.append(f"  prompt: {meta['positive']}")
                    if meta["negative"]:
                        parts.append(f"  negative: {meta['negative']}")
                    if meta["checkpoint"]:
                        parts.append(f"  checkpoint: {meta['checkpoint']}")
                    if meta["loras"]:
                        parts.append(f"  loras: {', '.join(meta['loras'])}")
                    if meta.get("sampler"):
                        s = meta["sampler"]
                        parts.append(f"  sampler: {', '.join(f'{k}={v}' for k,v in s.items() if k != 'seed')}")
                    if meta.get("resolution"):
                        parts.append(f"  resolution: {meta['resolution']}")
                    ref_parts.append("\n".join(parts))
                else:
                    ref_parts.append(f"  file: {fname} (no workflow metadata)")
            ref_block = "\n\n".join(ref_parts)

    # Build the full message — user text + reference appendix
    if ref_block:
        full_msg = f"{user_msg}\n\n---\n[Reference images:\n{ref_block}\n]"
    else:
        full_msg = user_msg

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    if not api_hist:
        api_hist = [{"role": "system", "content": system_prompt_for(model)}]

    api_hist = api_hist + [{"role": "user", "content": full_msg}]
    # Show the full enriched message in chat so user can verify
    display_hist = display_hist + [{"role": "user", "content": full_msg}]
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


# ---- Output helpers ----
def _list_outputs():
    """Return list of (abs_path, filename) newest first."""
    if not OUTPUT_DIR.exists():
        return []
    _MEDIA_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".gif")
    files = sorted(
        [p for p in OUTPUT_DIR.rglob("*") if p.suffix.lower() in _MEDIA_EXTS],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )[:60]
    return [(str(p), p.name) for p in files]

def refresh_gallery():
    return _list_outputs()

def _extract_png_meta(path):
    """Extract prompt, negative, checkpoint, loras, sampler from ComfyUI PNG metadata."""
    try:
        from PIL import Image
        img = Image.open(path)
        raw = img.info.get("prompt", "")
        if not raw:
            return None
        nodes = json.loads(raw)
        positive = negative = checkpoint = ""
        loras = []
        sampler = {}
        resolution = ""
        clip_texts = []  # collect all CLIP texts, sort later

        for nid, node in nodes.items():
            ct = node.get("class_type", "")
            inp = node.get("inputs", {})
            if ct == "CLIPTextEncode":
                txt = inp.get("text", "")
                if txt:
                    clip_texts.append((nid, txt))
            elif "CheckpointLoader" in ct:
                checkpoint = inp.get("ckpt_name", "")
            elif ct == "LoraLoader":
                ln = inp.get("lora_name", "")
                sw = inp.get("strength_model", "")
                if ln:
                    loras.append(f"{ln} ({sw})")
            elif "KSampler" in ct:
                for k in ["steps", "cfg", "sampler_name", "scheduler", "denoise", "seed"]:
                    if k in inp:
                        sampler[k] = inp[k]
            elif "LatentImage" in ct or "EmptyLatent" in ct:
                w, h = inp.get("width"), inp.get("height")
                if w and h:
                    resolution = f"{w}x{h}"

        # Sort CLIP texts into positive/negative
        NEG_WORDS = {"worst quality", "blurry", "deformed", "ugly", "bad anatomy",
                     "bad hands", "missing fingers", "extra fingers", "low quality",
                     "disfigured", "mutation", "watermark"}
        for nid, txt in clip_texts:
            txt_lower = txt.lower()
            # Check if it's a negative prompt (contains negative keywords or node ID hints)
            is_neg = (any(w in txt_lower for w in NEG_WORDS)
                      or "neg" in str(nid).lower())
            if is_neg:
                if not negative or len(txt) > len(negative):
                    negative = txt
            else:
                if not positive or len(txt) > len(positive):
                    positive = txt

        if positive or negative or checkpoint:
            return {"positive": positive, "negative": negative,
                    "checkpoint": checkpoint, "loras": loras,
                    "sampler": sampler, "resolution": resolution}

        # Fallback: A1111/Forge "parameters" format (used by CivitAI images)
        params_str = img.info.get("parameters", "")
        if params_str:
            return _parse_a1111_params(params_str)

        return None
    except Exception:
        return None


def _parse_a1111_params(params_str):
    """Parse A1111/Forge 'parameters' PNG metadata into our standard format."""
    import re as _re
    lines = params_str.strip().split("\n")
    positive = ""
    negative = ""
    checkpoint = ""
    loras = []
    sampler = {}
    resolution = ""

    # First line(s) before "Negative prompt:" are the positive prompt
    # Lines after "Negative prompt:" until the settings line are negative
    # Last line with "Steps: N, Sampler: ..." is the settings
    pos_parts = []
    neg_parts = []
    settings_line = ""
    in_negative = False

    for line in lines:
        if line.startswith("Negative prompt:"):
            in_negative = True
            neg_parts.append(line[len("Negative prompt:"):].strip())
        elif _re.match(r'^Steps:\s*\d', line):
            settings_line = line
        elif in_negative:
            neg_parts.append(line.strip())
        else:
            pos_parts.append(line.strip())

    positive = " ".join(pos_parts).strip()
    negative = " ".join(neg_parts).strip()

    # Extract LoRAs from prompt text: <lora:name:weight>
    for m in _re.finditer(r'<lora:([^:>]+):([^>]+)>', positive):
        loras.append(f"{m.group(1)} ({m.group(2)})")
    # Clean LoRA tags from prompt text
    positive = _re.sub(r',?\s*<lora:[^>]+>', '', positive).strip().rstrip(',').strip()

    # Parse settings line
    if settings_line:
        for kv in settings_line.split(","):
            kv = kv.strip()
            if ":" in kv:
                k, v = kv.split(":", 1)
                k, v = k.strip(), v.strip()
                if k == "Steps": sampler["steps"] = int(v)
                elif k == "Sampler": sampler["sampler_name"] = v
                elif k == "Schedule type": sampler["scheduler"] = v
                elif k == "CFG scale": sampler["cfg"] = float(v)
                elif k == "Seed": sampler["seed"] = int(v)
                elif k == "Size":
                    resolution = v
                elif k == "Model":
                    checkpoint = v

    return {"positive": positive, "negative": negative,
            "checkpoint": checkpoint, "loras": loras,
            "sampler": sampler, "resolution": resolution}

# Cache metadata to avoid re-reading PNGs on every 4s refresh
_meta_cache = {}

def refresh_output_html():
    """Render output strip as HTML cards with selectable thumbnails."""
    files = _list_outputs()
    if not files:
        return '<div style="color:#888;text-align:center;padding:40px;">No outputs yet</div>'
    cards = []
    for path, name in files:
        # Get or cache metadata
        if path not in _meta_cache:
            _meta_cache[path] = _extract_png_meta(path)
        meta = _meta_cache[path]

        details_html = ""
        if meta:
            parts = []
            if meta["positive"]:
                p = meta["positive"].replace("&", "&amp;").replace("<", "&lt;")
                parts.append(f"<b>Prompt:</b> {p}")
            if meta["negative"]:
                n = meta["negative"].replace("&", "&amp;").replace("<", "&lt;")
                parts.append(f"<b>Negative:</b> {n}")
            if meta["checkpoint"]:
                parts.append(f"<b>Checkpoint:</b> {meta['checkpoint']}")
            if meta["loras"]:
                parts.append(f"<b>LoRAs:</b> {', '.join(meta['loras'])}")
            if meta.get("sampler"):
                s = meta["sampler"]
                sampler_str = ", ".join(f"{k}={v}" for k, v in s.items() if k != "seed")
                parts.append(f"<b>Sampler:</b> {sampler_str}")
            if meta.get("resolution"):
                parts.append(f"<b>Resolution:</b> {meta['resolution']}")
            details_html = f'''<details style="margin-top:2px;" onclick="event.stopPropagation()">
  <summary style="font-size:10px;color:#666;cursor:pointer;list-style:none;display:flex;align-items:center;gap:3px;">
    <span style="font-size:7px;">&#9654;</span> workflow</summary>
  <div style="font-size:10px;color:#999;padding:4px 0 2px;line-height:1.5;user-select:all;">{"<br>".join(parts)}</div>
</details>'''

        is_video = Path(name).suffix.lower() in (".mp4", ".webm", ".gif")
        if is_video:
            media_html = f'''<video src="/gradio_api/file={path}" style="width:100%;border-radius:6px;display:block;"
       controls preload="metadata" onclick="event.stopPropagation()"></video>'''
        else:
            media_html = f'''<img src="/gradio_api/file={path}" style="width:100%;border-radius:6px;display:block;"
       onclick="event.stopPropagation();openLightbox(this.src)" loading="lazy">'''

        cards.append(f'''<div class="out-card" data-path="{name}" onclick="toggleSelect(this)"
             style="margin-bottom:10px;border:2px solid transparent;border-radius:8px;
                    padding:4px;cursor:pointer;transition:border-color 0.15s;">
  {media_html}
  <div style="display:flex;align-items:center;gap:4px;padding:3px 2px 0;">
    <input type="checkbox" class="out-check" onclick="event.stopPropagation();toggleSelect(this.closest('.out-card'))"
           style="margin:0;flex-shrink:0;accent-color:#4a9eff;">
    <span style="font-family:monospace;font-size:11px;color:#aaa;user-select:all;word-break:break-all;flex:1;min-width:0;">{name}</span>
    <span class="del-btn" onclick="event.stopPropagation();deleteFile('{name}')"
          style="flex-shrink:0;cursor:pointer;font-size:14px;opacity:0.4;padding:0 2px;"
          onmouseenter="this.style.opacity='1';this.style.color='#e55'"
          onmouseleave="this.style.opacity='0.4';this.style.color=''">&#128465;</span>
  </div>
  {details_html}
</div>''')
    return "\n".join(cards)

OUTPUT_STRIP_JS = """
<script>
function toggleSelect(card) {
  card.classList.toggle('selected');
  const on = card.classList.contains('selected');
  card.style.borderColor = on ? '#4a9eff' : 'transparent';
  const cb = card.querySelector('.out-check');
  if (cb) cb.checked = on;
  updateSelectedState();
}
function updateSelectedState() {
  const selected = [...document.querySelectorAll('.out-card.selected')].map(c => c.dataset.path);
  const el = document.querySelector('#selected-files-state textarea');
  if (el) { el.value = selected.join('\\n'); el.dispatchEvent(new Event('input', {bubbles:true})); }
}
function openLightbox(src) {
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.9);z-index:9999;display:flex;align-items:center;justify-content:center;cursor:zoom-out;';
  overlay.onclick = () => overlay.remove();
  const img = document.createElement('img');
  img.src = src;
  img.style.cssText = 'max-width:95vw;max-height:95vh;border-radius:8px;';
  overlay.appendChild(img);
  document.body.appendChild(overlay);
}
function deleteFile(name) {
  const card = document.querySelector('.out-card[data-path="' + name + '"]');
  if (card) { card.style.opacity = '0.3'; card.style.pointerEvents = 'none'; }
  fetch('/api/delete_output', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name: name})
  }).then(r => { if (card) card.remove(); })
    .catch(() => { if (card) { card.style.opacity = '1'; card.style.pointerEvents = ''; } });
}
function clearSelections() {
  document.querySelectorAll('.out-card.selected').forEach(c => {
    c.classList.remove('selected');
    c.style.borderColor = 'transparent';
    const cb = c.querySelector('.out-check');
    if (cb) cb.checked = false;
  });
  updateSelectedState();
}
</script>
"""

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

def delete_output_file(filename):
    """Delete a file from the output directory."""
    if not filename or not filename.strip():
        return ""
    name = filename.strip()
    target = OUTPUT_DIR / name
    if target.exists() and str(target).startswith(str(OUTPUT_DIR)):
        target.unlink()
        _meta_cache.pop(str(target), None)
    return ""

def add_reference_images(file_list):
    """Copy uploaded reference images into the ComfyUI output dir so they appear in the strip."""
    if not file_list:
        return None
    import shutil
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    added = []
    for f in file_list:
        src = Path(f) if isinstance(f, str) else Path(f.name)
        if not src.exists():
            continue
        dest_name = f"ref_{src.name}" if not src.name.startswith("ref_") else src.name
        dest = OUTPUT_DIR / dest_name
        shutil.copy2(str(src), str(dest))
        added.append(dest_name)
        # Invalidate cache so it shows on next refresh
        _meta_cache.pop(str(dest), None)
    return None  # clear the file picker after upload

saved_key = API_KEY_FILE.read_text().strip() if API_KEY_FILE.exists() else ""

# ---- UI ----
CSS = """
footer {display:none !important}
.gradio-container {max-width: 100% !important; padding: 8px !important;}
.gradio-container > .main, .gradio-container .contain {max-width: 100% !important; padding: 0 !important;}
.chatbot {height: calc(100vh - 320px) !important;}
#comfyframe iframe {width: 100%; height: 800px; border: 1px solid #444; border-radius: 8px;}
#output-strip {overflow-y: auto; max-height: calc(100vh - 100px);}
#output-html {overflow-y: auto; max-height: calc(100vh - 240px); padding-right: 4px;}
.out-card.selected {background: rgba(74,158,255,0.1);}
"""

with gr.Blocks(title="ComfyUI Studio", fill_height=True) as demo:
    api_state = gr.State([])

    with gr.Row():
        # Left: chat (60%)
        with gr.Column(scale=3, min_width=400):
            with gr.Row():
                gr.Markdown("### Studio")
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
            msg = gr.Textbox(placeholder="Describe what you want to generate...",
                             lines=2, show_label=False)
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary", scale=3)
                clear_btn = gr.Button("Clear", scale=1)

        # Right: output strip (40%)
        with gr.Column(scale=2, min_width=300, elem_id="output-strip"):
            with gr.Row():
                gr.Markdown("### Outputs")
                deselect_btn = gr.Button("Clear selection", size="sm", scale=0, min_width=100)
            ref_upload = gr.File(label="Add reference images", file_count="multiple",
                                 file_types=["image"], height=60)
            output_html = gr.HTML(value=refresh_output_html() + OUTPUT_STRIP_JS,
                                  elem_id="output-html")
            selected_state = gr.Textbox(value="", visible=False, elem_id="selected-files-state")
            delete_trigger = gr.Textbox(value="", visible=False, elem_id="delete-trigger")
            with gr.Row():
                refresh_btn = gr.Button("↻ Refresh", size="sm", scale=1)
                sync_btn = gr.Button("💾 N8Razer", size="sm", scale=1,
                                      variant="secondary")
                rescue_btn = gr.Button("🛟 Rescue", size="sm", scale=1,
                                        link="http://localhost:7681")
            sync_status = gr.Markdown("", visible=False)

    # ComfyUI workspace — collapsed by default
    with gr.Accordion("ComfyUI workspace", open=False):
        _comfy_host = _get_ts_hostname() or "localhost"
        _comfy_url = f"http://{_comfy_host}.tail2b8e3e.ts.net:8188" if _comfy_host and _comfy_host != "localhost" else "http://localhost:8188"
        gr.HTML(f'<iframe id="comfyframe" src="{_comfy_url}" allow="clipboard-write"'
                f' style="width:100%;height:800px;border:1px solid #444;border-radius:8px;"></iframe>')

    def _refresh_html():
        return refresh_output_html() + OUTPUT_STRIP_JS

    # Auto-refresh: poll outputs every 4s
    output_timer = gr.Timer(value=4, active=True)
    output_timer.tick(_refresh_html, None, output_html)

    # Wiring
    send_btn.click(chat, [msg, chatbot, api_state, api_key, model_pick, selected_state],
                   [msg, chatbot, api_state]).then(_refresh_html, None, output_html)
    msg.submit(chat, [msg, chatbot, api_state, api_key, model_pick, selected_state],
               [msg, chatbot, api_state]).then(_refresh_html, None, output_html)
    clear_btn.click(lambda: ([], []), None, [chatbot, api_state])
    refresh_btn.click(_refresh_html, None, output_html)
    ref_upload.change(add_reference_images, ref_upload, ref_upload).then(_refresh_html, None, output_html)
    delete_trigger.change(delete_output_file, delete_trigger, delete_trigger).then(_refresh_html, None, output_html)
    deselect_btn.click(lambda: "", None, selected_state, js="() => { clearSelections(); }")
    sync_btn.click(sync_to_n8razer, None, sync_status)
    api_key.change(save_key, api_key, None)
    edit_key_btn.click(lambda: gr.update(visible=True), None, api_key)

if __name__ == "__main__":
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn

    app = FastAPI()

    # Register custom API routes BEFORE mounting Gradio (which catches /)
    @app.post("/api/delete_output")
    async def api_delete_output(request: Request):
        data = await request.json()
        name = (data.get("name") or "").strip()
        if not name or ".." in name or "/" in name:
            return JSONResponse({"error": "invalid"}, status_code=400)
        target = OUTPUT_DIR / name
        if target.exists() and target.is_file():
            target.unlink()
            _meta_cache.pop(str(target), None)
            return JSONResponse({"status": "deleted"})
        return JSONResponse({"error": "not found"}, status_code=404)

    demo.queue()
    app = gr.mount_gradio_app(app, demo, path="/",
                               allowed_paths=[str(OUTPUT_DIR)])
    uvicorn.run(app, host="0.0.0.0", port=3000)

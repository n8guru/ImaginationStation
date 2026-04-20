"""HuggingFace Hub client — fetch model files and metadata."""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

HF_API = "https://huggingface.co/api/models"
HF_DL = "https://huggingface.co"


def fetch_model(repo_file, token=None):
    """Fetch metadata for a HuggingFace model file.

    repo_file: "org/repo/filename.safetensors" or "org/repo" (picks first .safetensors)

    Returns dict with: name, filename, download_url, base_model, category, source_id.
    """
    parts = repo_file.split("/", 2)
    if len(parts) < 2:
        raise ValueError(f"Expected 'org/repo' or 'org/repo/filename', got: {repo_file}")

    repo_id = f"{parts[0]}/{parts[1]}"
    filename = parts[2] if len(parts) > 2 else None

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Get repo info
    resp = httpx.get(f"{HF_API}/{repo_id}", headers=headers, timeout=30)
    resp.raise_for_status()
    repo_data = resp.json()

    # Find the file
    siblings = repo_data.get("siblings", [])
    model_files = [
        s for s in siblings
        if s["rfilename"].endswith((".safetensors", ".ckpt", ".pt", ".bin"))
    ]

    if filename:
        match = [s for s in model_files if s["rfilename"] == filename]
        if not match:
            available = [s["rfilename"] for s in model_files]
            raise ValueError(f"File '{filename}' not found in {repo_id}. Available: {available}")
        target = match[0]
    elif model_files:
        # Pick the largest .safetensors file (likely the main checkpoint)
        safetensors = [s for s in model_files if s["rfilename"].endswith(".safetensors")]
        target = safetensors[0] if safetensors else model_files[0]
    else:
        raise ValueError(f"No model files found in {repo_id}")

    target_filename = target["rfilename"]
    download_url = f"{HF_DL}/{repo_id}/resolve/main/{target_filename}"

    # Guess base model from tags or model name
    tags = repo_data.get("tags", [])
    base_model = _guess_base_model(tags, repo_id)

    # Guess category from filename/path
    category = _guess_category(target_filename)

    return {
        "name": repo_data.get("modelId", repo_id),
        "filename": target_filename.split("/")[-1],
        "download_url": download_url,
        "base_model": base_model,
        "category": category,
        "source": "huggingface",
        "source_id": f"{repo_id}/{target_filename}",
        "trigger_words": "",
        "preview_url": None,
    }


def download_file(url, dest_path, token=None):
    """Download a file from HuggingFace with progress logging."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with httpx.stream("GET", url, headers=headers, timeout=600,
                      follow_redirects=True) as resp:
        resp.raise_for_status()
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_bytes(65536):
                f.write(chunk)
                downloaded += len(chunk)
        logger.info("Downloaded %s (%d bytes)", dest_path, downloaded)
    return dest_path


def _guess_base_model(tags, repo_id):
    """Best-effort base model detection from HF tags."""
    tag_str = " ".join(tags).lower() + " " + repo_id.lower()
    if "flux" in tag_str:
        return "Flux"
    if "sdxl" in tag_str or "stable-diffusion-xl" in tag_str:
        return "SDXL"
    if "pony" in tag_str:
        return "Pony"
    if "sd-1" in tag_str or "stable-diffusion-v1" in tag_str or "sd1.5" in tag_str:
        return "SD1.5"
    return "unknown"


def _guess_category(filename):
    """Guess model category from filename/path."""
    fl = filename.lower()
    if "lora" in fl:
        return "lora"
    if "vae" in fl:
        return "vae"
    if "controlnet" in fl or "control" in fl:
        return "controlnet"
    if "embedding" in fl or "textual" in fl:
        return "embedding"
    if "upscale" in fl:
        return "upscaler"
    return "checkpoint"

"""CivitAI API client — fetch model metadata and download files."""

import logging
import os
import tempfile

import httpx

logger = logging.getLogger(__name__)

API_BASE = "https://civitai.com/api/v1"

# CivitAI type → our category mapping
TYPE_MAP = {
    "Checkpoint": "checkpoint",
    "LORA": "lora",
    "TextualInversion": "embedding",
    "Controlnet": "controlnet",
    "VAE": "vae",
    "Upscaler": "upscaler",
    "AestheticGradient": "style",
    "Hypernetwork": "style",
}

# CivitAI baseModel → our base_model mapping
BASE_MAP = {
    "SD 1.5": "SD1.5",
    "SD 1.4": "SD1.5",
    "SDXL 0.9": "SDXL",
    "SDXL 1.0": "SDXL",
    "SDXL Turbo": "SDXL",
    "SDXL Lightning": "SDXL",
    "Pony": "Pony",
    "Flux.1 D": "Flux",
    "Flux.1 S": "Flux",
}


def fetch_model(model_id, token=None):
    """Fetch model metadata from CivitAI.

    Returns dict with: name, type, category, base_model, trigger_words,
    download_url, filename, preview_url, source_id, version_name.
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"{API_BASE}/models/{model_id}"
    resp = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    data = resp.json()

    name = data.get("name", "Unknown")
    model_type = data.get("type", "LORA")
    category = TYPE_MAP.get(model_type, "style")

    # Use the first (latest) model version
    versions = data.get("modelVersions", [])
    if not versions:
        raise ValueError(f"Model {model_id} has no versions")
    ver = versions[0]

    base_model_raw = ver.get("baseModel", "")
    base_model = BASE_MAP.get(base_model_raw, base_model_raw or "unknown")
    trigger_words = ", ".join(ver.get("trainedWords", []))
    version_name = ver.get("name", "")

    # Find the primary model file
    files = ver.get("files", [])
    model_file = None
    for f in files:
        if f.get("type") == "Model" or f.get("name", "").endswith(".safetensors"):
            model_file = f
            break
    if not model_file and files:
        model_file = files[0]
    if not model_file:
        raise ValueError(f"Model {model_id} version {ver.get('id')} has no files")

    download_url = model_file.get("downloadUrl", ver.get("downloadUrl", ""))
    filename = model_file.get("name", f"civitai_{model_id}.safetensors")

    # SHA-256 from API (avoids downloading just to hash)
    hashes = model_file.get("hashes", {})
    sha256 = (hashes.get("SHA256") or "").lower()

    # Preview image — first non-NSFW image, or first image
    images = ver.get("images", [])
    preview_url = None
    for img in images:
        if not img.get("nsfw"):
            preview_url = img.get("url")
            break
    if not preview_url and images:
        preview_url = images[0].get("url")

    return {
        "name": name,
        "type": model_type,
        "category": category,
        "base_model": base_model,
        "trigger_words": trigger_words,
        "download_url": download_url,
        "filename": filename,
        "sha256": sha256,
        "preview_url": preview_url,
        "source": "civitai",
        "source_id": str(model_id),
        "version_name": version_name,
        "version_id": ver.get("id"),
    }


def download_file(url, dest_path, token=None):
    """Download a file from CivitAI with progress logging."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with httpx.stream("GET", url, headers=headers, timeout=600,
                      follow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_bytes(65536):
                f.write(chunk)
                downloaded += len(chunk)
        logger.info("Downloaded %s (%d bytes)", dest_path, downloaded)
    return dest_path


def download_preview(url, dest_path):
    """Download a preview image."""
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)
    return dest_path

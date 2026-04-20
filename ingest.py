#!/usr/bin/env python3
"""Model ingest CLI — add models to the library via CivitAI, HuggingFace, or local path.

Usage:
    python ingest.py civitai 288982
    python ingest.py hf stabilityai/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors
    python ingest.py local ./my_lora.safetensors --base sdxl --triggers "anime style" --category lora

All paths: sha256 dedupe → upload to Spaces → write manifest row → push manifest.
Preview images pulled during ingest when available (CivitAI, HF).

Requires: COMFY_S3_ENDPOINT, COMFY_S3_BUCKET, COMFY_S3_ACCESS_KEY, COMFY_S3_SECRET_KEY in env.
Optional: CIVITAI_API_TOKEN for gated CivitAI models.
"""

import logging
import os
import sys
import tempfile

import typer

# Add repo root to path for library imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from library.manifest import (
    open_db, add_model, get_by_sha256, hash_file,
    push_manifest, pull_manifest, upload_model_file, upload_preview,
    DEFAULT_DB_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("ingest")

app = typer.Typer(help="Model library ingest — add models to the manifest and Spaces.")


def _load_env():
    """Load .env if present."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        from dotenv import load_dotenv
        load_dotenv(env_path)


def _category_to_subdir(category):
    """Map manifest category to ComfyUI models/ subdirectory."""
    return {
        "checkpoint": "checkpoints",
        "lora": "loras",
        "vae": "vae",
        "controlnet": "controlnet",
        "embedding": "embeddings",
        "upscaler": "upscalers",
        "style": "loras",
    }.get(category, "other")


def _ingest_preview(preview_url, sha256, source_label):
    """Download and upload a preview image if URL is available."""
    if not preview_url:
        return ""
    try:
        import httpx
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        resp = httpx.get(preview_url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            f.write(resp.content)
        spaces_key = upload_preview(tmp_path, sha256)
        os.unlink(tmp_path)
        logger.info("Preview uploaded: %s", spaces_key)
        return spaces_key
    except Exception as e:
        logger.warning("Preview download failed for %s: %s", source_label, e)
        return ""


@app.command()
def civitai(
    model_id: int = typer.Argument(..., help="CivitAI model ID (from URL)"),
    db_path: str = typer.Option(str(DEFAULT_DB_PATH), help="Manifest DB path"),
):
    """Ingest a model from CivitAI by model ID."""
    _load_env()
    from library.civitai import fetch_model, download_file

    token = os.environ.get("CIVITAI_API_TOKEN", "")

    # Pull latest manifest
    logger.info("Pulling manifest from Spaces...")
    try:
        pull_manifest(db_path)
    except Exception as e:
        logger.warning("Could not pull manifest (starting fresh): %s", e)

    conn = open_db(db_path)

    # Fetch metadata
    logger.info("Fetching CivitAI model %d...", model_id)
    meta = fetch_model(model_id, token=token)
    logger.info("Found: %s (%s, %s)", meta["name"], meta["category"], meta["base_model"])

    # Download to temp
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name

    logger.info("Downloading %s...", meta["filename"])
    download_file(meta["download_url"], tmp_path, token=token)

    # SHA-256 dedupe
    sha = hash_file(tmp_path)
    existing = get_by_sha256(conn, sha)
    if existing:
        logger.info("DUPLICATE: sha256 %s already in manifest as '%s'. Skipping.",
                     sha[:12], existing["display_name"])
        os.unlink(tmp_path)
        conn.close()
        raise typer.Exit(0)

    # Upload to Spaces
    subdir = _category_to_subdir(meta["category"])
    spaces_key = f"models/{subdir}/{meta['filename']}"
    logger.info("Uploading to s3://%s...", spaces_key)
    upload_model_file(tmp_path, spaces_key)
    os.unlink(tmp_path)

    # Preview
    preview_path = _ingest_preview(meta["preview_url"], sha, f"civitai:{model_id}")

    # Display name: "Model Name — Version" if version differs from name
    display_name = meta["name"]
    if meta["version_name"] and meta["version_name"] != meta["name"]:
        display_name = f"{meta['name']} — {meta['version_name']}"

    # Write manifest row
    row_id = add_model(
        conn,
        filename=meta["filename"],
        sha256=sha,
        display_name=display_name,
        category=meta["category"],
        base_model=meta["base_model"],
        trigger_words=meta["trigger_words"],
        source="civitai",
        source_id=str(model_id),
        preview_path=preview_path,
        spaces_key=spaces_key,
        notes=f"CivitAI type: {meta['type']}",
    )

    if row_id:
        logger.info("Manifest row added (id=%d): %s", row_id, display_name)
        push_manifest(db_path)
        logger.info("Manifest pushed to Spaces.")
    else:
        logger.warning("Failed to add manifest row (sha256 conflict?)")

    conn.close()
    logger.info("Done: %s → %s", display_name, spaces_key)


@app.command()
def hf(
    repo_file: str = typer.Argument(..., help="HuggingFace repo/file (e.g. org/repo/file.safetensors)"),
    name: str = typer.Option("", help="Display name override"),
    base: str = typer.Option("", help="Base model override (SDXL, Flux, SD1.5, Pony)"),
    category: str = typer.Option("", help="Category override (checkpoint, lora, vae, etc.)"),
    triggers: str = typer.Option("", help="Trigger words"),
    db_path: str = typer.Option(str(DEFAULT_DB_PATH), help="Manifest DB path"),
):
    """Ingest a model from HuggingFace by repo/file path."""
    _load_env()
    from library.huggingface import fetch_model, download_file

    token = os.environ.get("HF_TOKEN", "")

    # Pull latest manifest
    logger.info("Pulling manifest from Spaces...")
    try:
        pull_manifest(db_path)
    except Exception as e:
        logger.warning("Could not pull manifest (starting fresh): %s", e)

    conn = open_db(db_path)

    # Fetch metadata
    logger.info("Fetching HuggingFace %s...", repo_file)
    meta = fetch_model(repo_file, token=token)
    logger.info("Found: %s (%s, %s)", meta["name"], meta["category"], meta["base_model"])

    # Apply overrides
    if name:
        meta["name"] = name
    if base:
        meta["base_model"] = base
    if category:
        meta["category"] = category
    if triggers:
        meta["trigger_words"] = triggers

    # Download to temp
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name

    logger.info("Downloading %s...", meta["filename"])
    download_file(meta["download_url"], tmp_path, token=token)

    # SHA-256 dedupe
    sha = hash_file(tmp_path)
    existing = get_by_sha256(conn, sha)
    if existing:
        logger.info("DUPLICATE: sha256 %s already in manifest as '%s'. Skipping.",
                     sha[:12], existing["display_name"])
        os.unlink(tmp_path)
        conn.close()
        raise typer.Exit(0)

    # Upload to Spaces
    subdir = _category_to_subdir(meta["category"])
    spaces_key = f"models/{subdir}/{meta['filename']}"
    logger.info("Uploading to s3://%s...", spaces_key)
    upload_model_file(tmp_path, spaces_key)
    os.unlink(tmp_path)

    # Preview
    preview_path = _ingest_preview(meta.get("preview_url"), sha, f"hf:{repo_file}")

    # Write manifest row
    row_id = add_model(
        conn,
        filename=meta["filename"],
        sha256=sha,
        display_name=meta["name"],
        category=meta["category"],
        base_model=meta["base_model"],
        trigger_words=meta["trigger_words"],
        source="huggingface",
        source_id=meta["source_id"],
        preview_path=preview_path,
        spaces_key=spaces_key,
    )

    if row_id:
        logger.info("Manifest row added (id=%d): %s", row_id, meta["name"])
        push_manifest(db_path)
        logger.info("Manifest pushed to Spaces.")
    else:
        logger.warning("Failed to add manifest row (sha256 conflict?)")

    conn.close()
    logger.info("Done: %s → %s", meta["name"], spaces_key)


@app.command()
def local(
    path: str = typer.Argument(..., help="Local file path"),
    name: str = typer.Option("", help="Display name (default: filename)"),
    base: str = typer.Option("unknown", "--base", help="Base model: SDXL, Flux, SD1.5, Pony"),
    triggers: str = typer.Option("", help="Trigger words (comma-separated)"),
    category: str = typer.Option("lora", help="Category: checkpoint, lora, vae, controlnet, embedding, style"),
    weight_range: str = typer.Option("0.6-0.8", help="Recommended weight range"),
    notes: str = typer.Option("", help="Freeform notes"),
    db_path: str = typer.Option(str(DEFAULT_DB_PATH), help="Manifest DB path"),
):
    """Ingest a local model file with manually supplied metadata."""
    _load_env()

    if not os.path.exists(path):
        logger.error("File not found: %s", path)
        raise typer.Exit(1)

    filename = os.path.basename(path)
    display_name = name or filename

    # Pull latest manifest
    logger.info("Pulling manifest from Spaces...")
    try:
        pull_manifest(db_path)
    except Exception as e:
        logger.warning("Could not pull manifest (starting fresh): %s", e)

    conn = open_db(db_path)

    # SHA-256 dedupe
    logger.info("Hashing %s...", filename)
    sha = hash_file(path)
    existing = get_by_sha256(conn, sha)
    if existing:
        logger.info("DUPLICATE: sha256 %s already in manifest as '%s'. Skipping.",
                     sha[:12], existing["display_name"])
        conn.close()
        raise typer.Exit(0)

    # Upload to Spaces
    subdir = _category_to_subdir(category)
    spaces_key = f"models/{subdir}/{filename}"
    logger.info("Uploading to s3://%s...", spaces_key)
    upload_model_file(path, spaces_key)

    # Write manifest row
    row_id = add_model(
        conn,
        filename=filename,
        sha256=sha,
        display_name=display_name,
        category=category,
        base_model=base,
        trigger_words=triggers,
        weight_range=weight_range,
        source="custom",
        notes=notes,
        spaces_key=spaces_key,
    )

    if row_id:
        logger.info("Manifest row added (id=%d): %s", row_id, display_name)
        push_manifest(db_path)
        logger.info("Manifest pushed to Spaces.")
    else:
        logger.warning("Failed to add manifest row (sha256 conflict?)")

    conn.close()
    logger.info("Done: %s → %s", display_name, spaces_key)


if __name__ == "__main__":
    app()

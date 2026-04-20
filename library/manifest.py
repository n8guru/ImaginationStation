"""Model library manifest — SQLite schema, CRUD, DO Spaces sync.

The manifest is the source of truth for the model library. Every model
file on Spaces MUST have a corresponding manifest row. Files that arrive
outside the ingest pipeline are flagged as orphans during reconciliation.

CONCURRENCY: V1 is single-writer. Do not run ingest from multiple
machines simultaneously — the SQLite file is synced to/from Spaces as a
whole, and concurrent writes will clobber. This is a known constraint,
not a bug. A Spaces object-lock lease pattern can fix it in V2.
"""

import hashlib
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_PATH = Path("/workspace/library/manifest.db")
SPACES_MANIFEST_KEY = "library/manifest.db"

# S3 config keys (same env vars as comfy_client.py)
_S3_ENV = {
    "endpoint": "COMFY_S3_ENDPOINT",
    "bucket": "COMFY_S3_BUCKET",
    "access_key": "COMFY_S3_ACCESS_KEY",
    "secret_key": "COMFY_S3_SECRET_KEY",
    "region": "COMFY_S3_REGION",
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS models (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL,
    sha256          TEXT NOT NULL UNIQUE,
    display_name    TEXT NOT NULL,
    category        TEXT NOT NULL DEFAULT 'style',
    base_model      TEXT NOT NULL DEFAULT 'unknown',
    trigger_words   TEXT DEFAULT '',
    weight_range    TEXT DEFAULT '0.6-0.8',
    source          TEXT DEFAULT 'custom',
    source_id       TEXT DEFAULT '',
    notes           TEXT DEFAULT '',
    preview_path    TEXT DEFAULT '',
    spaces_key      TEXT NOT NULL,
    date_added      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_models_sha256 ON models(sha256);
CREATE INDEX IF NOT EXISTS idx_models_base_model ON models(base_model);
CREATE INDEX IF NOT EXISTS idx_models_category ON models(category);
CREATE INDEX IF NOT EXISTS idx_models_filename ON models(filename);

CREATE VIRTUAL TABLE IF NOT EXISTS models_fts USING fts5(
    display_name, category, base_model, trigger_words, notes,
    content=models,
    content_rowid=id
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS models_ai AFTER INSERT ON models BEGIN
    INSERT INTO models_fts(rowid, display_name, category, base_model, trigger_words, notes)
    VALUES (new.id, new.display_name, new.category, new.base_model, new.trigger_words, new.notes);
END;

CREATE TRIGGER IF NOT EXISTS models_ad AFTER DELETE ON models BEGIN
    INSERT INTO models_fts(models_fts, rowid, display_name, category, base_model, trigger_words, notes)
    VALUES ('delete', old.id, old.display_name, old.category, old.base_model, old.trigger_words, old.notes);
END;

CREATE TRIGGER IF NOT EXISTS models_au AFTER UPDATE ON models BEGIN
    INSERT INTO models_fts(models_fts, rowid, display_name, category, base_model, trigger_words, notes)
    VALUES ('delete', old.id, old.display_name, old.category, old.base_model, old.trigger_words, old.notes);
    INSERT INTO models_fts(rowid, display_name, category, base_model, trigger_words, notes)
    VALUES (new.id, new.display_name, new.category, new.base_model, new.trigger_words, new.notes);
END;
"""


def _get_s3():
    """Return a boto3 S3 client configured for DO Spaces, or None."""
    import boto3
    endpoint = os.environ.get(_S3_ENV["endpoint"], "")
    access_key = os.environ.get(_S3_ENV["access_key"], "")
    secret_key = os.environ.get(_S3_ENV["secret_key"], "")
    region = os.environ.get(_S3_ENV["region"], "sfo3")

    if not all([endpoint, access_key, secret_key]):
        return None, None

    bucket = os.environ.get(_S3_ENV["bucket"], "imagination-models")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    return client, bucket


def init_db(db_path=None):
    """Initialize the manifest database. Idempotent."""
    db_path = Path(db_path or DEFAULT_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def open_db(db_path=None):
    """Open an existing manifest database."""
    db_path = Path(db_path or DEFAULT_DB_PATH)
    if not db_path.exists():
        return init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def add_model(conn, *, filename, sha256, display_name, category="style",
              base_model="unknown", trigger_words="", weight_range="0.6-0.8",
              source="custom", source_id="", notes="", preview_path="",
              spaces_key=""):
    """Insert a model row. Returns the new row id, or None if sha256 exists."""
    try:
        cur = conn.execute(
            """INSERT INTO models
               (filename, sha256, display_name, category, base_model,
                trigger_words, weight_range, source, source_id, notes,
                preview_path, spaces_key, date_added)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (filename, sha256, display_name, category, base_model,
             trigger_words, weight_range, source, source_id, notes,
             preview_path, spaces_key,
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        logger.warning("Model with sha256 %s already exists (dedupe)", sha256[:12])
        return None


def get_by_sha256(conn, sha256):
    """Look up a model by sha256. Returns dict or None."""
    row = conn.execute("SELECT * FROM models WHERE sha256 = ?", (sha256,)).fetchone()
    return dict(row) if row else None


def get_by_filename(conn, filename):
    """Look up a model by filename. Returns dict or None."""
    row = conn.execute("SELECT * FROM models WHERE filename = ?", (filename,)).fetchone()
    return dict(row) if row else None


def search(conn, query, base_model=None, category=None, limit=20):
    """Fuzzy search over the manifest using FTS5.

    Returns list of dicts with match rank.
    """
    where_parts = []
    params = []

    if query:
        where_parts.append("models_fts MATCH ?")
        # FTS5 query: wrap terms for prefix matching
        fts_query = " OR ".join(f'"{term}"*' for term in query.split())
        params.append(fts_query)

    sql = "SELECT m.*, rank FROM models_fts fts JOIN models m ON m.id = fts.rowid"
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)
    sql += " ORDER BY rank"
    sql += f" LIMIT {limit}"

    rows = conn.execute(sql, params).fetchall()
    results = [dict(r) for r in rows]

    # Post-filter by base_model and category (not in FTS)
    if base_model:
        bm = base_model.lower()
        results = [r for r in results if bm in r["base_model"].lower()]
    if category:
        cat = category.lower()
        results = [r for r in results if cat in r["category"].lower()]

    return results


def list_checkpoints(conn, base_model=None):
    """List checkpoint models, optionally filtered by base_model compatibility."""
    sql = "SELECT * FROM models WHERE category = 'checkpoint'"
    params = []
    if base_model:
        sql += " AND LOWER(base_model) LIKE ?"
        params.append(f"%{base_model.lower()}%")
    sql += " ORDER BY display_name"
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def list_all(conn):
    """List all models. Returns list of dicts."""
    return [dict(r) for r in conn.execute("SELECT * FROM models ORDER BY date_added DESC").fetchall()]


def hash_file(filepath, chunk_size=8192):
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Spaces sync
# ---------------------------------------------------------------------------

def push_manifest(db_path=None):
    """Upload the local manifest.db to DO Spaces."""
    db_path = Path(db_path or DEFAULT_DB_PATH)
    s3, bucket = _get_s3()
    if not s3:
        raise RuntimeError("S3 not configured — set COMFY_S3_* env vars")
    s3.upload_file(str(db_path), bucket, SPACES_MANIFEST_KEY)
    logger.info("Manifest pushed to s3://%s/%s", bucket, SPACES_MANIFEST_KEY)


def pull_manifest(db_path=None):
    """Download the manifest.db from DO Spaces. Creates parent dirs."""
    db_path = Path(db_path or DEFAULT_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    s3, bucket = _get_s3()
    if not s3:
        raise RuntimeError("S3 not configured — set COMFY_S3_* env vars")
    try:
        s3.download_file(bucket, SPACES_MANIFEST_KEY, str(db_path))
        logger.info("Manifest pulled from s3://%s/%s", bucket, SPACES_MANIFEST_KEY)
    except s3.exceptions.NoSuchKey:
        logger.info("No manifest on Spaces — initializing fresh")
        init_db(db_path)


def upload_model_file(local_path, spaces_key):
    """Upload a model file to DO Spaces from a local path."""
    s3, bucket = _get_s3()
    if not s3:
        raise RuntimeError("S3 not configured")
    logger.info("Uploading %s → s3://%s/%s", local_path, bucket, spaces_key)
    s3.upload_file(str(local_path), bucket, spaces_key)


def stream_upload(download_url, spaces_key, headers=None):
    """Stream a file from a URL directly to DO Spaces via S3 multipart upload.

    Uses chunked download + multipart upload to avoid buffering the whole
    file in memory. Handles multi-GB model files on low-RAM machines.
    """
    import httpx

    s3, bucket = _get_s3()
    if not s3:
        raise RuntimeError("S3 not configured")

    logger.info("Streaming %s → s3://%s/%s", download_url, bucket, spaces_key)

    CHUNK_SIZE = 16 * 1024 * 1024  # 16MB parts (DO Spaces minimum is 5MB)

    # Start multipart upload
    mpu = s3.create_multipart_upload(Bucket=bucket, Key=spaces_key)
    upload_id = mpu["UploadId"]

    parts = []
    part_num = 1
    total_bytes = 0

    try:
        with httpx.stream("GET", download_url, headers=headers or {},
                          timeout=600, follow_redirects=True) as resp:
            resp.raise_for_status()
            buf = b""
            for chunk in resp.iter_bytes(65536):
                buf += chunk
                while len(buf) >= CHUNK_SIZE:
                    part_data = buf[:CHUNK_SIZE]
                    buf = buf[CHUNK_SIZE:]
                    from io import BytesIO
                    r = s3.upload_part(
                        Bucket=bucket, Key=spaces_key,
                        UploadId=upload_id, PartNumber=part_num,
                        Body=BytesIO(part_data),
                    )
                    parts.append({"ETag": r["ETag"], "PartNumber": part_num})
                    total_bytes += len(part_data)
                    part_num += 1

            # Upload remaining bytes
            if buf:
                from io import BytesIO
                r = s3.upload_part(
                    Bucket=bucket, Key=spaces_key,
                    UploadId=upload_id, PartNumber=part_num,
                    Body=BytesIO(buf),
                )
                parts.append({"ETag": r["ETag"], "PartNumber": part_num})
                total_bytes += len(buf)

        # Complete multipart upload
        s3.complete_multipart_upload(
            Bucket=bucket, Key=spaces_key, UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
        logger.info("Streamed %d bytes (%d parts) to s3://%s/%s",
                     total_bytes, len(parts), bucket, spaces_key)
        return total_bytes

    except Exception:
        # Abort on failure to avoid orphaned parts
        s3.abort_multipart_upload(Bucket=bucket, Key=spaces_key, UploadId=upload_id)
        raise


def upload_preview(local_path, sha256):
    """Upload a preview image to DO Spaces at previews/<sha256>.jpg."""
    spaces_key = f"previews/{sha256}.jpg"
    s3, bucket = _get_s3()
    if not s3:
        raise RuntimeError("S3 not configured")
    s3.upload_file(str(local_path), bucket, spaces_key)
    return spaces_key


def reconcile(conn, models_root="/workspace/ComfyUI/models"):
    """Reconcile local models/ tree against manifest.

    Returns dict with:
        - orphans: files on disk without a manifest row
        - missing: manifest rows without a file on disk
    """
    models_root = Path(models_root)
    if not models_root.exists():
        return {"orphans": [], "missing": []}

    # Build set of all spaces_keys in manifest
    manifest_keys = set()
    rows = conn.execute("SELECT spaces_key, filename FROM models").fetchall()
    for r in rows:
        manifest_keys.add(r["spaces_key"])

    # Walk the models directory
    orphans = []
    disk_keys = set()
    for p in models_root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith(".") or p.name.startswith("put_"):
            continue
        rel = str(p.relative_to(models_root))
        spaces_key = f"models/{rel}"
        disk_keys.add(spaces_key)
        if spaces_key not in manifest_keys:
            orphans.append({"path": str(p), "spaces_key": spaces_key, "size": p.stat().st_size})

    # Check for manifest rows without a file on disk
    missing = []
    for r in rows:
        if r["spaces_key"] not in disk_keys:
            missing.append({"filename": r["filename"], "spaces_key": r["spaces_key"]})

    if orphans:
        logger.warning(
            "ORPHAN FILES: %d files in models/ without manifest rows. "
            "These were not ingested through the pipeline:",
            len(orphans),
        )
        for o in orphans:
            logger.warning("  orphan: %s (%d bytes)", o["spaces_key"], o["size"])

    if missing:
        logger.info("%d manifest entries without local files (not yet synced or removed)", len(missing))

    return {"orphans": orphans, "missing": missing}

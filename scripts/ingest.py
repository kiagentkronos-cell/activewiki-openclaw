#!/usr/bin/env python3
"""ActiveWiki: process files from inbox/<scope>/[subfolders]/ through Docling.

Scope (private|family|public) is the first path segment under inbox/. Files can
live at arbitrary depth inside the scope — the folder chain between scope and
file is captured as `inbox_path` in metadata.json and forwarded to the
destiller as a categorisation hint. processed/ and failed/ mirror the inbox
folder structure; sources/ stays flat per scope.

Configuration via activewiki.json (see activewiki.example.json).
"""
import hashlib
import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

# ── Config loading ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, wikis_root, scopes, docling_venv

_CONFIG = load_config()
_WIKIS_ROOT = wikis_root(_CONFIG)
_SCOPES = scopes(_CONFIG)
_DOCLING_VENV = docling_venv(_CONFIG)

# Make stdout/stderr tolerate filenames with non-UTF-8 bytes (surrogateescape).
sys.stdout.reconfigure(errors="backslashreplace")
sys.stderr.reconfigure(errors="backslashreplace")

# ── Paths (derived from config) ─────────────────────────────────────────────
INBOX = _WIKIS_ROOT / "inbox"
PROCESSING = _WIKIS_ROOT / "processing"
PROCESSED = _WIKIS_ROOT / "processed"
FAILED = _WIKIS_ROOT / "failed"
SOURCES = _WIKIS_ROOT / "sources"


def short_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def safe_fsname(s: str) -> str:
    """Normalize a filesystem name that may contain PEP 383 surrogate escapes
    from non-UTF-8 bytes on disk (typically Windows/CP1252 or Latin-1 copies)
    into clean UTF-8. Returns the string unchanged if already valid UTF-8."""
    try:
        s.encode("utf-8")
        return s
    except UnicodeEncodeError:
        raw = s.encode("utf-8", "surrogateescape")
        try:
            return raw.decode("cp1252")
        except UnicodeDecodeError:
            return raw.decode("latin-1")


def build_converter():
    """Build a Docling DocumentConverter.

    Imports are deferred so this module can be imported without docling installed
    (e.g. for --help or config validation).
    """
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        granite_picture_description,
    )
    from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.CUDA
    )
    opts.do_picture_description = True
    opts.picture_description_options = granite_picture_description.model_copy(
        update={
            "prompt": (
                "Beschreibe auf Deutsch kurz und sachlich, was auf dem Bild zu sehen ist. "
                "Nenne sichtbare Bauteile, Beschriftungen, Pfeile, Werte. "
                "Bei einfachen Icons oder abstrakten Symbolen: ein kurzer Satz reicht. "
                "Nur was konkret erkennbar ist — keine Wiederholungen, keine Vermutungen."
            ),
            "picture_area_threshold": 0.02,
            "generation_config": {"max_new_tokens": 300, "do_sample": False},
        }
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts),
        }
    )


def resolve_scope_and_path(src: Path) -> tuple[str, list[str]]:
    """Return (scope, inbox_path_segments) derived from src's location under INBOX.

    src must be a file under INBOX/<scope>/... — inbox_path_segments are the
    directory names between the scope dir and the file (empty if at scope root).
    """
    try:
        rel = src.resolve().relative_to(INBOX.resolve())
    except ValueError:
        raise ValueError(f"{src} is not under {INBOX}")
    parts = rel.parts
    if len(parts) < 2:
        raise ValueError(f"{src} has no scope segment under inbox/")
    scope = parts[0]
    if scope not in _SCOPES:
        raise ValueError(
            f"unknown scope {scope!r} for {src} — expected one of {SCOPES}"
        )
    inbox_path = list(parts[1:-1])
    return scope, inbox_path


def ingest(src: Path) -> Path:
    scope, inbox_path = resolve_scope_and_path(src)

    # Guard: files directly in scope root have no inbox_path → legacy mode → prompt explosion
    if not inbox_path:
        print(
            f"[warn] skipping {src.name}: files must be in a subfolder under inbox/{scope}/ "
            f"(not directly in inbox/{scope}/) — e.g. inbox/{scope}/thema/datei.pdf",
            file=sys.stderr,
        )
        return src  # file NOT processed; stays in inbox for manual re-filing

    sha = short_hash(src)
    source_id = f"{sha}-{safe_fsname(src.stem)[:40]}"
    safe_inbox_path = [safe_fsname(p) for p in inbox_path]
    safe_name = safe_fsname(src.name)
    out_dir = SOURCES / scope / source_id
    processed_subdir = PROCESSED / scope / Path(*inbox_path) if inbox_path else PROCESSED / scope
    processed_subdir.mkdir(parents=True, exist_ok=True)

    if out_dir.exists():
        dest = processed_subdir / src.name
        if src.resolve() != dest.resolve():
            shutil.move(str(src), str(dest))
        print(f"[skip] already ingested: {scope}/{source_id}")
        return out_dir

    PROCESSING.mkdir(exist_ok=True)
    # sha-prefix avoids collisions when files of the same basename live in
    # different inbox subfolders and happen to get queued together.
    work = PROCESSING / f"{sha}-{src.name}"
    shutil.move(str(src), str(work))

    try:
        result = build_converter().convert(str(work))
        doc = result.document

        out_dir.mkdir(parents=True, exist_ok=True)
        doc.save_as_markdown(out_dir / "document.md")
        doc.save_as_json(out_dir / "document.json")
        (out_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "source_id": source_id,
                    "scope": scope,
                    "original_name": safe_name,
                    "inbox_path": safe_inbox_path,
                    "sha256_prefix": sha,
                    "ingested_at": datetime.now(ZoneInfo("Europe/Berlin")).isoformat(),
                    "size_bytes": work.stat().st_size,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        shutil.move(str(work), str(processed_subdir / src.name))
        folder_label = "/".join(inbox_path) + "/" if inbox_path else ""
        print(f"[ok] {scope}/{folder_label}{src.name} -> sources/{scope}/{source_id}/")
        return out_dir

    except Exception:
        failed_subdir = FAILED / scope / Path(*inbox_path) if inbox_path else FAILED / scope
        failed_subdir.mkdir(parents=True, exist_ok=True)
        (failed_subdir / f"{src.name}.error.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        shutil.move(str(work), str(failed_subdir / src.name))
        folder_label = "/".join(inbox_path) + "/" if inbox_path else ""
        print(f"[fail] {scope}/{folder_label}{src.name} -> failed/{scope}/", file=sys.stderr)
        raise


def iter_inbox_files(scope_dir: Path):
    """Yield visible files under scope_dir recursively, skipping dotted paths."""
    for p in sorted(scope_dir.rglob("*")):
        if not p.is_file():
            continue
        if any(part.startswith(".") for part in p.relative_to(scope_dir).parts):
            continue
        yield p


def main() -> int:
    if len(sys.argv) > 1:
        target = Path(sys.argv[1]).resolve()
        if not target.is_file():
            print(f"not a file: {target}", file=sys.stderr)
            return 2
        ingest(target)
        return 0

    any_found = False
    for scope in _SCOPES:
        scope_dir = INBOX / scope
        if not scope_dir.exists():
            continue
        for f in iter_inbox_files(scope_dir):
            any_found = True
            ingest(f)
    if not any_found:
        print("inbox/ is empty")
    return 0


if __name__ == "__main__":
    sys.exit(main())

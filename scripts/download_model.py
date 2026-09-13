#!/usr/bin/env python3
"""Download the MiniMax-H3 files that native h3.c actually opens.

The Hugging Face repo is ~464 GB because it ships three copies of the same
weights:

* native ``FL2VA/`` (~134 GB) — required for t2va / first / last frame
* native ``Ref2VA/`` (~134 GB) — only the ~62 GB transformer differs
* root Diffusers layout (``transformer/``, ``transformer_ref/``,
  ``text_encoder/``, ``vae/``, …) — another ~196 GB, unused by h3.c

There are no extra quantizations in this repo. Community INT8/NVFP4 packs
live elsewhere and are not the h3.c baseline.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from h3_paths import default_model_dir  # noqa: E402

# Root-level Diffusers / docs. ``FL2VA/text_encoder/…`` does not match these.
DIFFUSERS_IGNORE = [
    "transformer/*",
    "transformer_ref/*",
    "text_encoder/*",
    "vae/*",
    "audio_vae/*",
    "assets/*",
    "docs/*",
    "scripts/*",
    "processor/*",
    "tokenizer/*",
    "scheduler/*",
    "audio_scheduler/*",
]

DIFFUSERS_ROOT_DIRS = (
    "transformer",
    "transformer_ref",
    "text_encoder",
    "vae",
    "audio_vae",
    "assets",
    "docs",
    "scripts",
    "processor",
    "tokenizer",
    "scheduler",
    "audio_scheduler",
)

# Byte-identical across FL2VA and Ref2VA (same git / LFS oids).
SHARED_COMPONENT_DIRS = (
    "text_encoder",
    "video_vae",
    "audio_vae",
    "tokenizer",
    "processor",
)

FL2VA_ALLOW = ["model_index.json", "FL2VA/*"]
REF2VA_TRANSFORMER_ALLOW = [
    "Ref2VA/model_index.json",
    "Ref2VA/transformer/*",
]

# Rough sizes from MiniMaxAI/MiniMax-H3 (Aug 2026). Dry-run prints live totals.
FL2VA_GIB = 134.1
REF2VA_TRANSFORMER_GIB = 61.7
FULL_REPO_GIB = 464.2
DIFFUSERS_GIB = 196.0


def allow_patterns(*, with_ref2va: bool) -> list[str]:
    allow = list(FL2VA_ALLOW)
    if with_ref2va:
        allow.extend(REF2VA_TRANSFORMER_ALLOW)
    return allow


def _filter_names(names: list[str], *, with_ref2va: bool) -> list[str]:
    from huggingface_hub.utils import filter_repo_objects

    return list(
        filter_repo_objects(
            names,
            allow_patterns=allow_patterns(with_ref2va=with_ref2va),
            ignore_patterns=DIFFUSERS_IGNORE,
        )
    )


def _is_empty_dir(path: Path) -> bool:
    if not path.is_dir() or path.is_symlink():
        return False
    try:
        next(path.iterdir())
    except StopIteration:
        return True
    return False


def _link_missing(src: Path, dst: Path, *, label: str) -> str | None:
    """Create a symlink only when dst has no files. Never delete weights."""
    if not src.exists():
        return None
    if dst.is_symlink():
        return f"keep {label}"
    if dst.exists() and not _is_empty_dir(dst):
        return f"keep existing {label}"
    if _is_empty_dir(dst):
        dst.rmdir()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(os.path.relpath(src, dst.parent), target_is_directory=src.is_dir())
    return f"linked {label} → {src.name}"


def reuse_existing_layout(dest: Path) -> list[str]:
    """Fill missing native paths from copies already on disk. Never deletes weights.

    Root ``text_encoder/`` / ``tokenizer/`` / ``processor/`` match the native
    trees. Root Diffusers ``transformer/`` and ``vae/`` do not, so those stay
    untouched and unused.
    """
    notes: list[str] = []
    fl = dest / "FL2VA"
    for name in ("text_encoder", "tokenizer", "processor"):
        root = dest / name
        msg = _link_missing(root, fl / name, label=f"FL2VA/{name}")
        if msg:
            notes.append(msg)
    rf = dest / "Ref2VA"
    if rf.exists():
        for name in SHARED_COMPONENT_DIRS:
            src = fl / name
            if not src.exists():
                src = dest / name
            msg = _link_missing(src, rf / name, label=f"Ref2VA/{name}")
            if msg:
                notes.append(msg)
    return notes


def link_shared_ref2va(dest: Path) -> list[str]:
    """Point missing Ref2VA encoder/VAE/tokenizer at FL2VA. Leaves existing copies."""
    return reuse_existing_layout(dest)


def _dir_bytes(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [d for d in dirs]
        for name in files:
            fp = Path(root) / name
            try:
                if fp.is_symlink():
                    continue
                total += fp.stat().st_size
            except OSError:
                continue
    return total


def _fmt_gib(n: int) -> str:
    return f" {n / (1024**3):.1f} GB"


def local_status(dest: Path) -> list[str]:
    if not dest.is_dir():
        return [f"missing {dest}"]
    lines = [f"{dest}  {_fmt_gib(_dir_bytes(dest)).strip()} on disk"]
    for name in (
        "FL2VA",
        "Ref2VA",
        *DIFFUSERS_ROOT_DIRS,
    ):
        path = dest / name
        if not path.exists():
            continue
        kind = "symlink" if path.is_symlink() else "dir"
        lines.append(f"  {name:16} {kind:7} {_fmt_gib(_dir_bytes(path))}")
    return lines


def _print_plan(*, dest: Path, with_ref2va: bool) -> None:
    extra = REF2VA_TRANSFORMER_GIB if with_ref2va else 0.0
    print("h3.c native MiniMax-H3 weights (BF16). Not Diffusers, not ComfyUI quants.")
    print(f"  destination:           {dest}")
    print(f"  FL2VA (required):      ~{FL2VA_GIB:.0f} GB")
    print(
        f"  Ref2VA transformer:    ~{REF2VA_TRANSFORMER_GIB:.0f} GB"
        + ("  [this run]" if with_ref2va else "  [skip — pass --with-ref2va]")
    )
    print(f"  this download:         ~{FL2VA_GIB + extra:.0f} GB")
    print(
        f"  skipped Diffusers copy: ~{DIFFUSERS_GIB:.0f} GB  "
        f"(root transformer/, text_encoder/, vae/, …)"
    )
    print(f"  full Hugging Face repo: ~{FULL_REPO_GIB:.0f} GB  (never fetch this)")
    print()


def summarize_selection(names: list[str], sizes: dict[str, int], *, with_ref2va: bool) -> tuple[list[str], int, int]:
    chosen = _filter_names(names, with_ref2va=with_ref2va)
    chosen_set = set(chosen)
    total = sum(sizes.get(n, 0) for n in chosen)
    skipped = sum(sz for n, sz in sizes.items() if n not in chosen_set)
    return chosen, total, skipped


def _print_selection(chosen: list[str], sizes: dict[str, int], *, n_repo: int, skipped: int) -> None:
    total = sum(sizes.get(n, 0) for n in chosen)
    print(f"would fetch {len(chosen)} files{_fmt_gib(total)}")
    print(f"would skip  {n_repo - len(chosen)} files{_fmt_gib(skipped)}")
    prefixes: dict[str, int] = {}
    for n in chosen:
        parts = n.split("/")
        key = (
            "/".join(parts[:2])
            if parts[0] in ("FL2VA", "Ref2VA") and len(parts) > 1
            else parts[0]
        )
        prefixes[key] = prefixes.get(key, 0) + sizes.get(n, 0)
    for key, sz in sorted(prefixes.items(), key=lambda kv: -kv[1]):
        print(f"  {key:28} {_fmt_gib(sz)}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download only the MiniMax-H3 files native h3.c loads"
    )
    p.add_argument("--repo", default="MiniMaxAI/MiniMax-H3")
    p.add_argument("--local-dir", type=Path, default=default_model_dir())
    p.add_argument(
        "--fl2va-only",
        action="store_true",
        help="Default. Kept for compatibility.",
    )
    p.add_argument(
        "--with-ref2va",
        action="store_true",
        help="Also fetch the Ref2VA transformer (~62 GB) and symlink shared encoder/VAE",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List remote files that would be downloaded and exit (metadata only)",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Show what is already on disk under --local-dir",
    )
    p.add_argument(
        "--prune",
        action="store_true",
        help=argparse.SUPPRESS,  # was destructive; kept so old commands fail safe
    )
    args = p.parse_args()
    dest = args.local_dir.expanduser().resolve()
    with_ref2va = bool(args.with_ref2va) and not args.fl2va_only

    if args.prune:
        raise SystemExit(
            "--prune is disabled: already-downloaded files are never deleted.\n"
            "Use --status to see what is on disk, then run a normal download to resume.\n"
            "Root Diffusers leftovers stay until you remove them yourself."
        )

    if args.status:
        print("\n".join(local_status(dest)))
        return

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        from h3_bootstrap import ensure_python_requirements

        ensure_python_requirements()
        from huggingface_hub import HfApi, snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    reused = reuse_existing_layout(dest)
    for line in reused:
        print(line)
    _print_plan(dest=dest, with_ref2va=with_ref2va)

    if args.dry_run:
        info = HfApi().model_info(args.repo, files_metadata=True)
        names = [s.rfilename for s in info.siblings]
        sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
        chosen, _total, skipped = summarize_selection(
            names, sizes, with_ref2va=with_ref2va
        )
        _print_selection(chosen, sizes, n_repo=len(names), skipped=skipped)
        return

    print(f"Downloading {args.repo} → {dest}")
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(dest),
        allow_patterns=allow_patterns(with_ref2va=with_ref2va),
        ignore_patterns=DIFFUSERS_IGNORE,
    )
    leftover = reuse_existing_layout(dest)
    for line in leftover:
        print(f"     {line}")

    fl = dest / "FL2VA"
    if not fl.is_dir():
        raise SystemExit(f"Download finished but {fl} is missing")
    for required in (
        "transformer/config.json",
        "tokenizer/tokenizer.json",
        "text_encoder",
        "transformer",
        "video_vae/source",
        "audio_vae",
    ):
        path = fl / required
        if not path.exists():
            raise SystemExit(f"Download finished but {path} is missing")
    print(f"OK: {dest}")
    print("     FL2VA present")
    if with_ref2va:
        tr = dest / "Ref2VA" / "transformer"
        if not tr.is_dir():
            print("     warning: Ref2VA transformer missing")
        else:
            print("     Ref2VA transformer present (shared encoder/VAE kept or linked)")
    else:
        print("     Ref2VA skipped (text/first/last-frame still work; pass --with-ref2va later)")


if __name__ == "__main__":
    main()

"""H3 DiT LoRA catalog, Hugging Face download, and h3.c --lora wiring."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from h3_paths import REPO_ROOT

log = logging.getLogger("h3")

TUTU_REPO = "tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA"
TUTU_GUIDANCE = (
    "Tutu 20→8 NFE LoRA for FL2VA. Trained for 8 Euler steps at strength 0.8. "
    "h3.c uses its own 8-step shifted schedule (not ComfyUI ManualSigmas). "
    "SSD streaming is off while a LoRA is enabled. Rebuild h3 after pull "
    "(scripts/build_h3.sh) so --lora is fused at DiT load."
)

BUILTIN_LORAS: list[dict[str, Any]] = [
    {
        "id": "tutu_20to8_nfe_step100",
        "label": "Tutu 20→8 NFE (step 100)",
        "spec": (
            f"https://huggingface.co/{TUTU_REPO}/resolve/main/"
            "comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000100-bf16-comfyui.safetensors"
        ),
        "scale": 0.8,
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "guidance": TUTU_GUIDANCE,
    },
    {
        "id": "tutu_20to8_nfe_step200",
        "label": "Tutu 20→8 NFE (step 200)",
        "spec": (
            f"https://huggingface.co/{TUTU_REPO}/resolve/main/"
            "comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000200-bf16-comfyui.safetensors"
        ),
        "scale": 0.8,
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "guidance": TUTU_GUIDANCE,
    },
    {
        "id": "tutu_20to8_nfe_step300",
        "label": "Tutu 20→8 NFE (step 300)",
        "spec": (
            f"https://huggingface.co/{TUTU_REPO}/resolve/main/"
            "comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000300-bf16-comfyui.safetensors"
        ),
        "scale": 0.8,
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "guidance": TUTU_GUIDANCE,
    },
]


def lora_cache_dir() -> Path:
    env = os.environ.get("H3_LORA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return REPO_ROOT / "models" / "loras"


def normalize_lora_spec(spec: str) -> str:
    raw = (spec or "").strip().strip("'\"")
    if not raw:
        return ""
    raw = raw.replace("/blob/", "/resolve/", 1)
    if raw.startswith("hf.co/"):
        raw = "https://huggingface.co/" + raw[len("hf.co/") :]
    if raw.startswith("huggingface.co/"):
        raw = "https://" + raw
    parsed = urlparse(raw)
    if parsed.netloc in {"hf.co", "www.hf.co"}:
        raw = f"https://huggingface.co{parsed.path}"
        if parsed.query:
            raw += f"?{parsed.query}"
    if re.fullmatch(r"[\w.-]+/[\w.-]+", raw):
        raw = f"https://huggingface.co/{raw}/resolve/main"
    return raw


def _parse_hf_resolve(url: str) -> tuple[str, str, str] | None:
    parsed = urlparse(url)
    if "huggingface.co" not in parsed.netloc:
        return None
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) >= 5 and parts[2] == "resolve":
        repo = f"{parts[0]}/{parts[1]}"
        revision = parts[3]
        filename = "/".join(parts[4:])
        return repo, revision, filename
    # Handle /repo/name/resolve/revision without filename
    if len(parts) == 4 and parts[2] == "resolve":
        return f"{parts[0]}/{parts[1]}", parts[3], ""
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}", "main", ""
    return None


def _label_for_spec(spec: str) -> str:
    parsed = _parse_hf_resolve(spec)
    if parsed and parsed[2]:
        return Path(parsed[2]).name
    path = Path(spec)
    if path.suffix:
        return path.name
    return spec[-48:] if len(spec) > 48 else spec


def _is_usable(path: Path | None) -> bool:
    return bool(path and path.is_file() and path.stat().st_size > 64)


def resolve_lora_path(spec: str) -> Path:
    """Local path or Hugging Face download into models/loras/."""
    spec = normalize_lora_spec(spec)
    if not spec:
        raise ValueError("LoRA spec is empty")
    local = Path(spec).expanduser()
    if local.is_file():
        return local.resolve()
    parsed = _parse_hf_resolve(spec)
    if parsed is None:
        if spec.startswith(("http://", "https://")):
            raise ValueError(f"only Hugging Face LoRA URLs are supported: {spec}")
        raise FileNotFoundError(f"LoRA file not found: {spec}")
    repo, revision, filename = parsed
    if not filename:
        filename = Path(BUILTIN_LORAS[0]["spec"]).name
        if repo == TUTU_REPO:
            filename = (
                "comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000100-"
                "bf16-comfyui.safetensors"
            )
    dest_dir = lora_cache_dir() / repo.replace("/", "__")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(filename).name
    if _is_usable(dest):
        return dest
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download LoRAs") from exc
    log.info("Downloading LoRA %s (%s) …", repo, filename)
    local_path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        revision=revision,
        local_dir=str(dest_dir),
    )
    path = Path(local_path).resolve()
    if path != dest and _is_usable(path) and not _is_usable(dest):
        try:
            dest.unlink(missing_ok=True)
            dest.symlink_to(path)
        except OSError:
            return path
    if not _is_usable(path):
        raise RuntimeError(f"downloaded LoRA is empty: {path}")
    return path


def lora_cached_path(spec: str) -> Path | None:
    spec = normalize_lora_spec(spec)
    local = Path(spec).expanduser()
    if local.is_file():
        return local.resolve()
    parsed = _parse_hf_resolve(spec)
    if parsed is None:
        return None
    repo, _rev, filename = parsed
    if not filename:
        return None
    dest = lora_cache_dir() / repo.replace("/", "__") / Path(filename).name
    return dest if _is_usable(dest) else None


def ensure_lora(spec: str) -> dict[str, Any]:
    normalized = normalize_lora_spec(spec)
    cached = lora_cached_path(normalized)
    if cached is not None:
        return {"ok": True, "spec": normalized, "path": str(cached), "cached": True}
    path = resolve_lora_path(normalized)
    return {"ok": True, "spec": normalized, "path": str(path), "cached": False}


def read_custom_loras(output_dir: Path) -> list[dict[str, Any]]:
    from web_ui import read_web_settings

    raw = read_web_settings(output_dir).get("custom_loras")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        spec = normalize_lora_spec(str(item.get("spec") or ""))
        lid = str(item.get("id") or "").strip()
        if not spec or not lid:
            continue
        try:
            scale = float(item.get("scale", 1.0))
        except (TypeError, ValueError):
            scale = 1.0
        out.append(
            {
                "id": lid,
                "label": str(item.get("label") or "").strip() or _label_for_spec(spec),
                "spec": spec,
                "scale": scale,
                "custom": True,
            }
        )
    return out


def write_custom_loras(output_dir: Path, entries: list[dict[str, Any]]) -> None:
    from web_ui import read_web_settings, write_web_settings

    data = read_web_settings(output_dir)
    data["custom_loras"] = [
        {
            "id": e["id"],
            "label": e.get("label") or _label_for_spec(str(e.get("spec") or "")),
            "spec": normalize_lora_spec(str(e.get("spec") or "")),
            "scale": float(e.get("scale") or 1.0),
        }
        for e in entries
        if e.get("id") and e.get("spec")
    ]
    write_web_settings(output_dir, data)


def lora_catalog(output_dir: Path | None = None) -> list[dict[str, Any]]:
    presets = [dict(item) for item in BUILTIN_LORAS]
    seen = {str(p["id"]) for p in presets}
    if output_dir is not None:
        for entry in read_custom_loras(output_dir):
            if entry["id"] in seen:
                continue
            cached = lora_cached_path(entry["spec"])
            entry["cached"] = cached is not None
            presets.append(entry)
            seen.add(entry["id"])
    for preset in presets:
        if "cached" not in preset:
            preset["cached"] = lora_cached_path(str(preset["spec"])) is not None
    return presets


def materialize_loras(specs: list[tuple[str, float]]) -> list[dict[str, Any]]:
    """Download each spec and return ``{spec, path, scale}`` dicts."""
    out: list[dict[str, Any]] = []
    for spec, scale in specs:
        path = resolve_lora_path(spec)
        out.append({"spec": spec, "path": str(path), "scale": float(scale)})
    return out


def parse_lora_specs(raw: Any, catalog: list[dict[str, Any]] | None = None) -> list[tuple[str, float]]:
    """Body `loras` / `lora_specs`: [{id, spec, scale}] or [[spec, scale], ...]."""
    if raw is None:
        return []
    items = raw
    if isinstance(raw, dict):
        items = [raw]
    if not isinstance(items, list):
        raise ValueError("loras must be a list")
    catalog = catalog or []
    by_id = {str(p.get("id")): p for p in catalog}
    out: list[tuple[str, float]] = []
    for item in items:
        spec = ""
        scale = 1.0
        if isinstance(item, dict):
            lid = str(item.get("id") or "").strip()
            if lid and lid in by_id:
                spec = str(by_id[lid].get("spec") or "")
                scale = float(item.get("scale", by_id[lid].get("scale", 1.0)))
            else:
                spec = str(item.get("spec") or item.get("path") or "")
                scale = float(item.get("scale", 1.0))
        elif isinstance(item, (list, tuple)) and item:
            spec = str(item[0])
            scale = float(item[1]) if len(item) > 1 else 1.0
        elif isinstance(item, str):
            spec = item
        spec = normalize_lora_spec(spec)
        if not spec:
            continue
        out.append((spec, max(0.0, float(scale))))
    return out

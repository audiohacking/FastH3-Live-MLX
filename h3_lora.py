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
CATALOG_SOURCE = "https://www.stablediffusiontutorials.com/2026/08/minimax-h3-lora-models.html"
LIGHTX2V_REPO = "lightx2v/Minimax-h3-Turbo"
LARRYVRH_REPO = "larryvrh/MiniMax-H3-Turbo-Lora"


def _hf(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def _entry(
    *,
    lid: str,
    label: str,
    spec: str,
    scale: float = 1.0,
    category: str = "Style",
    guidance: str = "",
    steps: int | None = None,
    layers: int | None = None,
    reuse: int | None = None,
    trigger: str | None = None,
    compatible: bool = True,
    source_url: str | None = None,
    repo: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": lid,
        "label": label,
        "spec": spec,
        "scale": scale,
        "category": category,
        "guidance": guidance,
        "compatible": compatible,
        "source_url": source_url
        or (f"https://huggingface.co/{repo}" if repo else CATALOG_SOURCE),
    }
    if steps is not None:
        out["steps"] = steps
        out["layers"] = layers if layers is not None else 50
        out["reuse"] = reuse if reuse is not None else 1
    if trigger:
        out["trigger"] = trigger
    return out


BUILTIN_LORAS: list[dict[str, Any]] = [
    _entry(
        lid="tutu_20to8_nfe_step100",
        label="Tutu 20→8 NFE (step 100)",
        spec=_hf(
            TUTU_REPO,
            "comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000100-bf16-comfyui.safetensors",
        ),
        scale=0.8,
        category="Speed",
        steps=8,
        repo=TUTU_REPO,
        guidance=TUTU_GUIDANCE,
    ),
    _entry(
        lid="tutu_20to8_nfe_step200",
        label="Tutu 20→8 NFE (step 200)",
        spec=_hf(
            TUTU_REPO,
            "comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000200-bf16-comfyui.safetensors",
        ),
        scale=0.8,
        category="Speed",
        steps=8,
        repo=TUTU_REPO,
        guidance=TUTU_GUIDANCE,
    ),
    _entry(
        lid="tutu_20to8_nfe_step300",
        label="Tutu 20→8 NFE (step 300)",
        spec=_hf(
            TUTU_REPO,
            "comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000300-bf16-comfyui.safetensors",
        ),
        scale=0.8,
        category="Speed",
        steps=8,
        repo=TUTU_REPO,
        guidance=TUTU_GUIDANCE,
    ),
    _entry(
        lid="lightx2v_fl2v_turbo_8step_v10",
        label="MiniMax-H3 FL2VA Turbo 8-step v1.0 (LightX2V)",
        spec=_hf(LIGHTX2V_REPO, "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
        scale=0.6,
        category="Speed",
        steps=8,
        repo=LIGHTX2V_REPO,
        guidance=(
            "LightX2V FL2VA 8-step turbo. Typical Continuity strength 0.6; "
            "video shift 6 / audio shift 3 when using that recipe."
        ),
    ),
    _entry(
        lid="lightx2v_fl2v_turbo_4step_v12_768p",
        label="MiniMax-H3 FL2VA Turbo 4-step v1.2 · 768p (LightX2V)",
        spec=_hf(
            LIGHTX2V_REPO,
            "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors",
        ),
        scale=0.6,
        category="Speed",
        steps=4,
        repo=LIGHTX2V_REPO,
        guidance="LightX2V FL2VA 4-step · 768p. Strength 0.6; shift 6/3 in Continuity graphs.",
    ),
    _entry(
        lid="lightx2v_ref2v_turbo_8step_v10_768p",
        label="MiniMax-H3 Ref2VA Turbo 8-step v1.0 · 768p (LightX2V)",
        spec=_hf(
            LIGHTX2V_REPO,
            "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
        ),
        scale=0.6,
        category="Speed",
        steps=8,
        repo=LIGHTX2V_REPO,
        guidance=(
            "LightX2V Ref2VA 8-step · 768p. Continuity turbo strength 0.6 with "
            "shift 6/3 and euler + beta."
        ),
    ),
    _entry(
        lid="larryvrh_turbo_v4_step600_ema",
        label="larryvrh Turbo v4 (step 600 EMA)",
        spec=_hf(LARRYVRH_REPO, "minimax_h3_turbo_v4_step600_ema.safetensors"),
        scale=1.0,
        category="Speed",
        steps=6,
        repo=LARRYVRH_REPO,
        guidance=(
            "larryvrh's current pick. Strength 1.0; 4 steps is the floor and "
            "6–8 looks better. Runs on h3's own 12/3 clocks."
        ),
    ),
    _entry(
        lid="official_turbo_4step_ckpt850",
        label="larryvrh Turbo v1 (ckpt 850)",
        spec=_hf(LARRYVRH_REPO, "minimax_h3_turbo_4step_ema_ckpt850.safetensors"),
        scale=1.0,
        category="Speed",
        steps=4,
        repo=LARRYVRH_REPO,
        guidance="Superseded v1 line — prefer v4 step 600 EMA for most work.",
    ),
    _entry(
        lid="fasth3_preview_rank64",
        label="FastH3 Dense 4-step (rank 64 preview)",
        spec=_hf(
            "drozbay/MiniMax-H3-FastH3-Preview-LoRA",
            "loras/minimax_h3_fl2va_fasth3_preview_v0.2_lora_pruned_rank64_fp16.safetensors",
        ),
        scale=1.0,
        category="Speed",
        steps=4,
        repo="drozbay/MiniMax-H3-FastH3-Preview-LoRA",
        guidance="Pruned FastH3 4-step preview. Official recipe is 4 Euler steps.",
    ),
    _entry(
        lid="fasth3_v1_dense_datafree",
        label="FastH3 v1 dense datafree (ComfyUI)",
        spec=_hf(
            "zerubroberts/MiniMax-H3-FastH3-v1-dense-datafree-ComfyUI",
            "fasth3_v1_dense_datafree_comfyui.safetensors",
        ),
        scale=1.0,
        category="Speed",
        steps=4,
        repo="zerubroberts/MiniMax-H3-FastH3-v1-dense-datafree-ComfyUI",
        guidance="FastH3 v1 dense datafree ComfyUI conversion. Try 4 Euler steps.",
    ),
    _entry(
        lid="realism_people",
        label="Realism People",
        spec=_hf(
            "fal/MiniMax-H3-Realism-People-LoRA",
            "h3-realism-people-t2v-i2v-r2v.safetensors",
        ),
        scale=1.0,
        category="Style",
        trigger="r34l1sm",
        repo="fal/MiniMax-H3-Realism-People-LoRA",
        guidance="Faces, skin, and documentary motion. Lead the prompt with r34l1sm. Scale 0.6–1.0.",
    ),
    _entry(
        lid="studio_1939_strong",
        label="Studio 1939 Animation (strong)",
        spec=_hf(
            "lovis93/studio-1939-old-animation-lora-minimax-h3",
            "studio1939-strong.safetensors",
        ),
        scale=0.85,
        category="Style",
        repo="lovis93/studio-1939-old-animation-lora-minimax-h3",
        guidance="1930s gouache / celluloid animation look.",
    ),
    _entry(
        lid="studio_1939_light",
        label="Studio 1939 Animation (light)",
        spec=_hf(
            "lovis93/studio-1939-old-animation-lora-minimax-h3",
            "studio1939-light.safetensors",
        ),
        scale=0.75,
        category="Style",
        repo="lovis93/studio-1939-old-animation-lora-minimax-h3",
        guidance="Lighter 1930s animation wash. Stacks less aggressively than Strong.",
    ),
    _entry(
        lid="vh5tape_vhs",
        label="VH5Tape VHS",
        spec=_hf("KennethFal/vh5tape-vhs-lora-minimax-h3", "vh5tape-comfyui.safetensors"),
        scale=0.8,
        category="Style",
        repo="KennethFal/vh5tape-vhs-lora-minimax-h3",
        guidance="1980s broadcast-to-VHS: chroma bleed, tracking noise, head-switching bands.",
    ),
    _entry(
        lid="lineart_anime",
        label="Anime line-art colorization",
        spec=_hf("DiffSynth-Studio/MiniMax-H3-LoRA-LineartAnime", "model.safetensors"),
        scale=1.0,
        category="Style",
        repo="DiffSynth-Studio/MiniMax-H3-LoRA-LineartAnime",
        guidance="Colorize anime line-art video while keeping the original lines.",
    ),
    _entry(
        lid="camera_motion_v1",
        label="Camera Motion",
        spec=_hf(
            "Jojocodex/minimax-h3-Camera-Motion-lora",
            "camera_motion_h3_lora_v1_3000_pruned.safetensors",
        ),
        scale=0.8,
        category="Motion",
        repo="Jojocodex/minimax-h3-Camera-Motion-lora",
        guidance="Push, pull, pan, tilt, orbit, handheld. Name the move in the prompt.",
    ),
    _entry(
        lid="spatial_physics_v2",
        label="Spatial Physics",
        spec=_hf(
            "Jojocodex/minimax-h3-spatial-physics-lora",
            "wushu_spatial_physics_v2_1000_pruned.safetensors",
        ),
        scale=0.8,
        category="Motion",
        repo="Jojocodex/minimax-h3-spatial-physics-lora",
        guidance="Collisions, stacking, falling, occlusion. Helps contact feel physical.",
    ),
    _entry(
        lid="wushu_action_v7_fl2va",
        label="Wushu Action v7 (FL2VA INT8)",
        spec=_hf(
            "Jojocodex/wushu-action-v7-minimax-h3-fl2va-ref2va-lora",
            "wushu_action_v7_fl2va_aitoolkit_adaln_full-int8convrot_bf16te_2000step.safetensors",
        ),
        scale=0.85,
        category="Motion",
        repo="Jojocodex/wushu-action-v7-minimax-h3-fl2va-ref2va-lora",
        guidance="Kung-fu / Wushu for the Comfy-Org INT8 convrot FL2VA DiT.",
    ),
    _entry(
        lid="equirect_360_v2",
        label="Equirectangular 360°",
        spec=_hf(
            "shamanic/minimax-h3-equi360-lora",
            "h3-equi360-reviewed-v2-step2500.safetensors",
        ),
        scale=1.0,
        category="Immersive",
        trigger="equirect360",
        repo="shamanic/minimax-h3-equi360-lora",
        guidance="Full spherical 360° frame. Lead with equirect360. Prefer a wide canvas.",
    ),
    _entry(
        lid="vr180_sbs_v2",
        label="VR180 side-by-side",
        spec=_hf("rehan-fal/minimax-h3-vr180-sbs-lora", "h3-vr180-sbs-lora-v2.safetensors"),
        scale=1.0,
        category="Immersive",
        repo="rehan-fal/minimax-h3-vr180-sbs-lora",
        guidance="Stereoscopic VR180 SBS (left | right). Wide landscape canvas.",
    ),
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


def catalog_entry(lora_id: str, output_dir: Path | None = None) -> dict[str, Any] | None:
    for item in lora_catalog(output_dir):
        if str(item.get("id")) == lora_id:
            return item
    return None


def lora_progress_bytes(spec: str) -> int:
    """Bytes already on disk for this spec (complete file or resume leftovers)."""
    spec = normalize_lora_spec(spec)
    cached = lora_cached_path(spec)
    if cached is not None:
        try:
            return cached.stat().st_size
        except OSError:
            return 0
    parsed = _parse_hf_resolve(spec)
    if parsed is None:
        return 0
    repo, _rev, filename = parsed
    if not filename:
        return 0
    dest_dir = lora_cache_dir() / repo.replace("/", "__")
    name = Path(filename).name
    total = 0
    candidates = [
        dest_dir / name,
        dest_dir / f"{name}.incomplete",
    ]
    if dest_dir.is_dir():
        candidates.extend(dest_dir.rglob("*.incomplete"))
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            total = max(total, resolved.stat().st_size)
        except OSError:
            continue
    return total


def lora_catalog(output_dir: Path | None = None) -> list[dict[str, Any]]:
    presets = [dict(item) for item in BUILTIN_LORAS]
    seen = {str(p["id"]) for p in presets}
    if output_dir is not None:
        for entry in read_custom_loras(output_dir):
            if entry["id"] in seen:
                continue
            cached = lora_cached_path(entry["spec"])
            entry["cached"] = cached is not None
            entry.setdefault("category", "Custom")
            entry.setdefault("compatible", True)
            presets.append(entry)
            seen.add(entry["id"])
    for preset in presets:
        if "cached" not in preset:
            spec = str(preset.get("spec") or "")
            preset["cached"] = bool(spec) and lora_cached_path(spec) is not None
        preset.setdefault("compatible", True)
        preset.setdefault("category", "Style")
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

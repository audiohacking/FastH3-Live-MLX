"""Random scene × cast pool (ported from fasth3-live PromptPool)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from h3_live import DEFAULT_CHARACTERS, DEFAULT_SCENES


class PromptPool:
    """Random scene × random character.

    Scene files use ``{NAME}`` / ``{NAME2}``… placeholders. Multiple scene files
    are drawn as one flat pool (a file's share of draws is its share of blocks).
    """

    def __init__(
        self,
        scenes_paths: list[Path] | None = None,
        characters_path: Path | None = None,
        curated_share: float = 0.30,
        *,
        explicit: bool = False,
    ) -> None:
        paths = list(scenes_paths) if scenes_paths else list(DEFAULT_SCENES)
        self.scenes_paths = [Path(p) for p in paths]
        self.explicit = explicit
        self.curated_share = float(curated_share)
        self._missing_warned: set[str] = set()
        chars = Path(characters_path) if characters_path else DEFAULT_CHARACTERS
        pools = json.loads(chars.read_text(encoding="utf-8"))
        self.curated: list[str] = list(pools.get("curated") or [])
        self.full: list[str] = list(pools.get("full") or [])
        if not self.full:
            raise ValueError(f"no characters in {chars}")

    def counts(self) -> dict[str, int]:
        return {str(p): len(self._blocks(p)) for p in self.scenes_paths}

    def _blocks(self, path: Path) -> list[str]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            if self.explicit:
                raise FileNotFoundError(f"--scenes {path}: {exc}") from exc
            key = str(path)
            if key not in self._missing_warned:
                self._missing_warned.add(key)
            return []
        return [b.strip() for b in text.split("\n---\n") if b.strip()]

    def scenes(self) -> list[str]:
        out: list[str] = []
        for p in self.scenes_paths:
            out.extend(self._blocks(p))
        if not out:
            raise RuntimeError("no scenes to draw from; check --scenes")
        return out

    def _pick(self) -> str:
        pool = (
            self.curated
            if (self.curated and random.random() < self.curated_share)
            else self.full
        )
        return random.choice(pool)

    def fill_names(self, scene: str) -> tuple[str, str]:
        """Replace ``{NAME}`` / ``{NAME2}``… slots; return ``(filled, cast_label)``."""
        slots = ["{NAME}"] + [f"{{NAME{i}}}" for i in range(2, 10)]
        slots = [s for s in slots if s in scene]
        picked: list[str] = []
        for _ in slots:
            name = self._pick()
            for _ in range(20):
                if name not in picked:
                    break
                name = self._pick()
            picked.append(name)
        for slot, name in sorted(zip(slots, picked), key=lambda p: -len(p[0])):
            scene = scene.replace(slot, name)
        return scene, " + ".join(picked) if picked else "(custom)"

    def draw(self) -> tuple[str, str, int]:
        """Return ``(filled_prompt, cast_label, scene_index)``."""
        scenes = self.scenes()
        idx = random.randrange(len(scenes))
        filled, cast = self.fill_names(scenes[idx])
        return filled, cast, idx

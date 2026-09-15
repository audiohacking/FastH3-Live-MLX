"""Random scene × cast pool (ported from fasth3-live PromptPool)."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from h3_live import (
    DATA_DIR,
    DEFAULT_CHARACTERS,
    DEFAULT_SCENES,
    LIVE_CURATED_SHARE,
    LIVE_ENSEMBLE_BIAS,
    LIVE_MAX_CAST,
)

# Short The Office phrases for <5s clips (lip-sync friendly).
DEFAULT_LINES = DATA_DIR / "h3_office_lines.json"
_QUOTE_KEYS = ["{QUOTE}"] + [f"{{QUOTE{i}}}" for i in range(2, 10)]


class PromptPool:
    """Random scene × random character.

    Scene files use ``{NAME}`` / ``{NAME2}``… placeholders. Optional ``{QUOTE}`` /
    ``{QUOTE2}``… are filled with a short character-matched Office line after cast
    is chosen. Cast is capped at ``max_cast`` slots (default ``LIVE_MAX_CAST`` = 3).
    """

    def __init__(
        self,
        scenes_paths: list[Path] | None = None,
        characters_path: Path | None = None,
        curated_share: float = LIVE_CURATED_SHARE,
        *,
        ensemble_only: bool = False,
        ensemble_bias: float = LIVE_ENSEMBLE_BIAS,
        max_cast: int = LIVE_MAX_CAST,
        explicit: bool = False,
        lines_path: Path | None = None,
    ) -> None:
        paths = list(scenes_paths) if scenes_paths else list(DEFAULT_SCENES)
        self.scenes_paths = [Path(p) for p in paths]
        self.explicit = explicit
        self.curated_share = float(curated_share)
        self.ensemble_only = bool(ensemble_only)
        self.ensemble_bias = max(0.0, min(1.0, float(ensemble_bias)))
        self.max_cast = max(1, int(max_cast))
        self._missing_warned: set[str] = set()
        self._cast_counts: dict[str, int] = {}
        chars = Path(characters_path) if characters_path else DEFAULT_CHARACTERS
        pools = json.loads(chars.read_text(encoding="utf-8"))
        self.curated: list[str] = list(pools.get("curated") or [])
        self.full: list[str] = list(pools.get("full") or [])
        if not self.full:
            raise ValueError(f"no characters in {chars}")
        lines_file = Path(lines_path) if lines_path else DEFAULT_LINES
        self.lines: dict[str, list[str]] = {}
        if lines_file.is_file():
            raw = json.loads(lines_file.read_text(encoding="utf-8"))
            self.lines = {
                str(k): [str(x) for x in v if str(x).strip()]
                for k, v in raw.items()
                if isinstance(v, list)
            }

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

    @staticmethod
    def all_slot_keys(scene: str) -> list[str]:
        """Every ``{NAME}`` / ``{NAME2}``… placeholder present (any count)."""
        keys = ["{NAME}"] + [f"{{NAME{i}}}" for i in range(2, 10)]
        return [s for s in keys if s in scene]

    def slot_keys(self, scene: str) -> list[str]:
        """Placeholders that will be filled, capped at ``max_cast``."""
        return self.all_slot_keys(scene)[: self.max_cast]

    @classmethod
    def slot_count(cls, scene: str) -> int:
        return len(cls.all_slot_keys(scene))

    @classmethod
    def is_ensemble(cls, scene: str) -> bool:
        """True for 3-hand scenes within the default Live cast cap."""
        n = cls.slot_count(scene)
        return 3 <= n <= LIVE_MAX_CAST

    def scenes(self) -> list[str]:
        out: list[str] = []
        for p in self.scenes_paths:
            out.extend(self._blocks(p))
        out = [s for s in out if self.slot_count(s) <= self.max_cast]
        if not out:
            raise RuntimeError("no scenes to draw from; check --scenes")
        if self.ensemble_only:
            ens = [
                s
                for s in out
                if self.slot_count(s) >= 3 and self.slot_count(s) <= self.max_cast
            ]
            if not ens:
                raise RuntimeError("ensemble-only set but no 3-character scenes found")
            return ens
        return out

    def _pick_unique(self, k: int) -> list[str]:
        """Pick ``k`` distinct names with usage-fair weights (underused cast rises)."""
        prefer_curated = bool(self.curated) and random.random() < self.curated_share
        primary = list(dict.fromkeys(self.curated if prefer_curated else self.full))
        fallback = list(dict.fromkeys(self.full))
        picked: list[str] = []
        for _ in range(k):
            pool = [n for n in primary if n not in picked] or [
                n for n in fallback if n not in picked
            ]
            if not pool:
                raise RuntimeError(
                    f"need {k} unique cast names but only "
                    f"{len(picked)} available after depleting the character pools"
                )
            weights = [1.0 / ((1 + self._cast_counts.get(n, 0)) ** 2) for n in pool]
            choice = random.choices(pool, weights=weights, k=1)[0]
            picked.append(choice)
            self._cast_counts[choice] = self._cast_counts.get(choice, 0) + 1
        return picked

    def pick_line(self, name: str) -> str:
        """Random short Office line for ``name`` (empty if unknown / no bank)."""
        bank = self.lines.get(name) or []
        if not bank:
            return ""
        return random.choice(bank)

    def _fill_quotes(self, scene: str, picked: list[str]) -> str:
        """Replace ``{QUOTE}`` / ``{QUOTE2}``… with a line matching each cast name."""
        for i, name in enumerate(picked):
            key = "{QUOTE}" if i == 0 else f"{{QUOTE{i + 1}}}"
            if key not in scene:
                continue
            line = self.pick_line(name)
            if not line:
                scene = re.sub(
                    rf",?\s*who (?:says|whispers|murmurs):\s*<d>\[English\]\s*{re.escape(key)}\s*</d>",
                    "",
                    scene,
                )
                scene = scene.replace(key, "…")
            else:
                scene = scene.replace(key, line)
        for key in _QUOTE_KEYS:
            if key in scene:
                scene = scene.replace(key, "…")
        return scene

    @staticmethod
    def _hardcoded_cast(scene: str) -> list[str]:
        """Names already written into a scene, in speak-order."""
        found = re.findall(
            r"([A-Z][A-Za-z.'\-]+(?: [A-Z][A-Za-z.'\-]+){0,3}) from The Office \(S\d+\)",
            scene,
        )
        out: list[str] = []
        for n in found:
            if n not in out:
                out.append(n)
        return out

    def fill_names(self, scene: str) -> tuple[str, str]:
        """Replace cast + quote slots; return ``(filled, cast_label)``.

        ``{NAME}``… get distinct cast. ``{QUOTE}``… get a short line from that
        character's Office phrase bank (for <5s lip-sync).
        """
        slots = self.slot_keys(scene)
        if not slots:
            hardcoded = self._hardcoded_cast(scene)
            for name in hardcoded:
                self._cast_counts[name] = self._cast_counts.get(name, 0) + 1
            scene = self._fill_quotes(scene, hardcoded)
            return scene, " + ".join(hardcoded) if hardcoded else "(no cast)"

        picked = self._pick_unique(len(slots))
        for slot, name in sorted(zip(slots, picked), key=lambda p: -len(p[0])):
            scene = scene.replace(slot, name)
        scene = self._fill_quotes(scene, picked)
        assert len(picked) == len(set(picked)), picked
        assert "{QUOTE}" not in scene
        return scene, " + ".join(picked)

    @staticmethod
    def looks_like_context_ir(text: str) -> bool:
        t = text.lower()
        return "integrated_multimodal_description:" in t

    @classmethod
    def wrap_idea_as_live_prompt(cls, idea: str) -> str:
        """Turn a freeform idea into the FastH3 Live Context-IR block.

        The random pool always emits full IR (description + soundscape + music).
        Bare one-liners are off-distribution for the student and hallucinate.
        Keeps the user's subject/action wording; does not invent dialogue.
        """
        idea = " ".join(idea.strip().split())
        if not idea:
            raise ValueError("empty idea")
        if not idea[0].isupper():
            idea = idea[0].upper() + idea[1:]
        if not idea.endswith("."):
            idea = idea + "."
        return (
            "integrated_multimodal_description: [Shot 1] Live-action, cinematic, "
            f"a medium group shot frames {idea} Soft key light, shallow depth of field, "
            "natural colour. The camera pushes in with small amplitude at slow speed. "
            "[Shot 2] At 00:08.000, the shot cuts to a close-up on the primary subject "
            "continuing the same beat with matching lighting.\n"
            "overall_soundscape: Diegetic ambience matching the scene over soft room tone; "
            "no crowd walla unless implied by the action.\n"
            "non_diegetic_music: N/A"
        )

    def normalize_custom(self, text: str) -> tuple[str, str]:
        """Prepare a dashboard custom prompt the same way as a pool scene."""
        text = text.strip()
        if not text:
            raise ValueError("empty custom prompt")
        if not self.looks_like_context_ir(text):
            text = self.wrap_idea_as_live_prompt(text)
        return self.fill_names(text)

    def draw(self) -> tuple[str, str, int]:
        """Return ``(filled_prompt, cast_label, scene_index)``."""
        scenes = self.scenes()
        if not self.ensemble_only and self.ensemble_bias > 0:
            ensembles = [i for i, s in enumerate(scenes) if self.is_ensemble(s)]
            if ensembles and random.random() < self.ensemble_bias:
                idx = random.choice(ensembles)
                filled, cast = self.fill_names(scenes[idx])
                return filled, cast, idx
        idx = random.randrange(len(scenes))
        filled, cast = self.fill_names(scenes[idx])
        return filled, cast, idx

"""Random scene × cast pool (ported from fasth3-live PromptPool)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from h3_live import (
    DEFAULT_CHARACTERS,
    DEFAULT_SCENES,
    LIVE_CURATED_SHARE,
    LIVE_ENSEMBLE_BIAS,
    LIVE_MAX_CAST,
)


class PromptPool:
    """Random scene × random character.

    Scene files use ``{NAME}`` / ``{NAME2}``… placeholders. Multiple scene files
    are drawn as one flat pool (a file's share of draws is its share of blocks).
    Cast is capped at ``max_cast`` slots (default ``LIVE_MAX_CAST`` = 3).
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
    ) -> None:
        paths = list(scenes_paths) if scenes_paths else list(DEFAULT_SCENES)
        self.scenes_paths = [Path(p) for p in paths]
        self.explicit = explicit
        self.curated_share = float(curated_share)
        self.ensemble_only = bool(ensemble_only)
        self.ensemble_bias = max(0.0, min(1.0, float(ensemble_bias)))
        self.max_cast = max(1, int(max_cast))
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
        # Drop blocks that ask for more cast than the Live cap (keeps pool finite).
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

    def _pick(self) -> str:
        pool = (
            self.curated
            if (self.curated and random.random() < self.curated_share)
            else self.full
        )
        return random.choice(pool)

    def fill_names(self, scene: str) -> tuple[str, str]:
        """Replace ``{NAME}`` / ``{NAME2}``… slots; return ``(filled, cast_label)``.

        At most ``max_cast`` names are filled. Every slot gets a distinct name —
        never the same character twice in one scene.
        """
        slots = self.slot_keys(scene)
        if not slots:
            return scene, "(no cast)"

        prefer_curated = bool(self.curated) and random.random() < self.curated_share
        primary = list(dict.fromkeys(self.curated if prefer_curated else self.full))
        fallback = list(dict.fromkeys(self.full))

        picked: list[str] = []
        for _ in slots:
            pool = [n for n in primary if n not in picked]
            if not pool:
                pool = [n for n in fallback if n not in picked]
            if not pool:
                raise RuntimeError(
                    f"need {len(slots)} unique cast names but only "
                    f"{len(picked)} available after depleting the character pools"
                )
            picked.append(random.choice(pool))

        for slot, name in sorted(zip(slots, picked), key=lambda p: -len(p[0])):
            scene = scene.replace(slot, name)
        assert len(picked) == len(set(picked)), picked
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
        """Prepare a dashboard custom prompt the same way as a pool scene.

        Full Context-IR is filled for ``{NAME}`` slots as-is. Short freeform
        ideas are wrapped into the Live IR skeleton first.
        """
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

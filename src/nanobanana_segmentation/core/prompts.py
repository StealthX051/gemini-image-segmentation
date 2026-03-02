from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional

import yaml


@dataclass(frozen=True)
class PromptAttempt:
    prompt_id: str
    prompt_text: str
    attempt_mode: str


DEFAULT_TEMPLATE_TEXT: Dict[str, str] = {
    "chromakey_v1": (
        "Task: Return a segmentation-mask surrogate image for the target structure only.\n"
        "Target: {target}\n"
        "Output requirements:\n"
        "1) Output image dimensions must exactly match input dimensions.\n"
        "2) Two colors only: background exactly #00FF00 and ROI exactly #FFFFFF.\n"
        "3) No text, no overlays, no transparency, no gradients, no blur, no antialiasing.\n"
        "4) Return only the mask surrogate image.\n"
        "{profile_addendum}\n"
        "{constraint_text}\n"
        "{tool_addendum}\n"
        "{model_addendum}"
    ),
    "chromakey_v2_strict": (
        "Correction pass. Regenerate the mask surrogate and strictly enforce all constraints.\n"
        "Target: {target}\n"
        "Hard requirements:\n"
        "- Exact input resolution\n"
        "- Background every pixel must be #00FF00\n"
        "- ROI every pixel must be #FFFFFF\n"
        "- Exactly two colors in output\n"
        "- No labels/text/graphics\n"
        "If any requirement cannot be met, still output the best compliant two-color mask image.\n"
        "{profile_addendum}\n"
        "{constraint_text}\n"
        "{tool_addendum}\n"
        "{model_addendum}"
    ),
    "bw_v1_fallback": (
        "Fallback mode: output a pure black-white binary mask image only.\n"
        "Target: {target}\n"
        "Hard requirements:\n"
        "- Exact input resolution\n"
        "- ROI white (#FFFFFF), background black (#000000)\n"
        "- No gray values, no antialiasing, no text/overlay\n"
        "{profile_addendum}\n"
        "{constraint_text}\n"
        "{tool_addendum}\n"
        "{model_addendum}"
    ),
}


def _profile_addendum(task_profile: str) -> str:
    profile = (task_profile or "blob").strip().lower()
    if profile == "thin":
        return (
            "Continuity constraint: preserve connected thin structures; avoid broken segments and speckle islands."
        )
    if profile == "low_contrast":
        return (
            "Low-contrast constraint: avoid scattered micro-islands; prefer one contiguous ROI when clinically plausible."
        )
    return "Blob profile: delineate the full visible target extent with a tight but complete boundary."


def _constraint_text(constraints: Dict[str, object] | None) -> str:
    if not constraints:
        return "No additional area/component constraints."
    lines: List[str] = []
    if constraints.get("min_area_frac") is not None:
        lines.append(f"Minimum ROI area fraction: {constraints['min_area_frac']}")
    if constraints.get("max_area_frac") is not None:
        lines.append(f"Maximum ROI area fraction: {constraints['max_area_frac']}")
    if constraints.get("single_component"):
        lines.append("ROI should be a single connected component.")
    if constraints.get("allow_border_touch") is False:
        lines.append("ROI should not touch image border.")
    if constraints.get("min_components") is not None:
        lines.append(f"Minimum connected components: {constraints['min_components']}")
    if constraints.get("max_components") is not None:
        lines.append(f"Maximum connected components: {constraints['max_components']}")
    return "\n".join(lines) if lines else "No additional area/component constraints."


def _tool_addendum(tool_mode: str) -> str:
    mode = (tool_mode or "closed").strip().lower()
    if mode == "text":
        return "Tool mode: text grounding enabled. You may use web text search, but do not reproduce source images."
    if mode == "image":
        return "Tool mode: image grounding enabled. Use external examples only as conceptual guidance, not as masks."
    if mode == "text_image":
        return "Tool mode: text + image grounding enabled. Use retrieval for reasoning only; output must remain original mask surrogate pixels."
    return "Tool mode: closed. Do not rely on external retrieval tools."


def _model_addendum(model_style: str) -> str:
    style = (model_style or "nanobanana_v1").strip().lower()
    if style == "nanobanana_v1":
        return (
            "NanoBanana image-generation mode: return one generated image only, with no markdown, JSON, or explanatory text."
        )
    return "Return one generated image only, without additional text."


def _normalize_template_map(raw: Mapping[str, object] | None) -> Dict[str, str]:
    merged: Dict[str, str] = dict(DEFAULT_TEMPLATE_TEXT)
    if not raw:
        return merged
    for key, value in raw.items():
        if isinstance(value, str) and key in merged:
            merged[key] = value
    return merged


def load_prompt_templates(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not path.exists():
        return dict(DEFAULT_TEMPLATE_TEXT)
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    templates = payload.get("templates") if isinstance(payload, dict) else None
    return _normalize_template_map(templates if isinstance(templates, dict) else None)


def _render_template(template: str, context: Mapping[str, str]) -> str:
    rendered = template.format(**context)
    lines = [line.rstrip() for line in rendered.splitlines()]
    compact = "\n".join(line for line in lines if line.strip())
    return compact.strip()


def build_retry_ladder_prompts(
    *,
    target: str,
    task_profile: str,
    constraints: Dict[str, object] | None,
    max_attempts: int,
    tool_mode: str = "closed",
    model_style: str = "nanobanana_v1",
    templates: Mapping[str, str] | None = None,
) -> List[PromptAttempt]:
    template_map = _normalize_template_map(dict(templates) if templates is not None else None)
    context = {
        "target": target,
        "profile_addendum": _profile_addendum(task_profile),
        "constraint_text": _constraint_text(constraints),
        "tool_addendum": _tool_addendum(tool_mode),
        "model_addendum": _model_addendum(model_style),
    }

    ladder = [
        PromptAttempt(
            prompt_id="chromakey_v1",
            prompt_text=_render_template(template_map["chromakey_v1"], context),
            attempt_mode="chromakey",
        ),
        PromptAttempt(
            prompt_id="chromakey_v2_strict",
            prompt_text=_render_template(template_map["chromakey_v2_strict"], context),
            attempt_mode="chromakey",
        ),
        PromptAttempt(
            prompt_id="bw_v1_fallback",
            prompt_text=_render_template(template_map["bw_v1_fallback"], context),
            attempt_mode="bw",
        ),
    ]
    return ladder[: max(1, int(max_attempts))]

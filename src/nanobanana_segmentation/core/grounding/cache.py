from __future__ import annotations

from pathlib import Path

from gemini_segmentation.cache import DiskRequestCache, build_request_cache_key


def build_grounding_cache_key(
    *,
    image_path: Path,
    model_name: str,
    prompt_hash: str,
    tool_mode: str,
    thinking_level: str,
) -> str:
    return build_request_cache_key(
        image_path=image_path,
        provider="nanobanana",
        model_name=model_name,
        prompt_hash=prompt_hash,
        prompt_family=tool_mode,
        temperature=None,
        thinking_budget=None,
        targets=(thinking_level,),
    )


__all__ = ["DiskRequestCache", "build_grounding_cache_key"]

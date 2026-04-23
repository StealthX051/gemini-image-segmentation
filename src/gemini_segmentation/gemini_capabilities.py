from __future__ import annotations


GEMINI_2_5_FLASH = "gemini-2.5-flash"
GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
GEMINI_2_5_PRO = "gemini-2.5-pro"
GEMINI_ROBOTICS_ER_1_5_PREVIEW = "gemini-robotics-er-1.5-preview"
GEMINI_ROBOTICS_ER_1_6_PREVIEW = "gemini-robotics-er-1.6-preview"


def normalize_gemini_model_name(model_name: str) -> str:
    if model_name.startswith("models/"):
        return model_name[len("models/") :]
    return model_name


def gemini_supports_explicit_cache(model_name: str) -> bool:
    return normalize_gemini_model_name(model_name) != GEMINI_ROBOTICS_ER_1_5_PREVIEW


def gemini_supports_structured_output(model_name: str) -> bool:
    return normalize_gemini_model_name(model_name) in {
        GEMINI_2_5_FLASH,
        GEMINI_2_5_FLASH_LITE,
        GEMINI_2_5_PRO,
    }


def gemini_supports_agentic_vision(model_name: str) -> bool:
    return normalize_gemini_model_name(model_name) == GEMINI_ROBOTICS_ER_1_6_PREVIEW

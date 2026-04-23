"""Provider-aware prompt construction utilities for segmentation tasks."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Mapping, Sequence

GEMINI_SEGMENTATION_PROMPT_SKELETON = (
    "Give the segmentation masks for the objects. "
    'Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d", '
    'the segmentation mask in key "mask", and the text label in the key "label". '
    "Use descriptive labels. "
    'The "mask" value must be a base64 encoded png returned as a PNG data URI beginning with "data:image/png;base64,". '
    "Do not return SVG path data, polygon coordinate lists, token strings, or other vector encodings."
)

# Backward-compatible alias used by existing tests and internal imports.
SCHEMA_PREAMBLE = GEMINI_SEGMENTATION_PROMPT_SKELETON

GEMINI_SEGMENTATION_RESPONSE_JSON_SCHEMA: Dict[str, object] = {
    "type": "array",
    "description": "Segmentation masks for the requested objects.",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "box_2d": {
                "type": "array",
                "description": "Normalized bounding box in [y0, x0, y1, x1] coordinates scaled between 0 and 1000.",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "integer"},
            },
            "mask": {
                "type": "string",
                "description": 'A base64 encoded png mask returned as a PNG data URI beginning with "data:image/png;base64,".',
            },
            "label": {
                "type": "string",
                "description": "Descriptive text label for the segmented object.",
            },
        },
        "required": ["box_2d", "mask", "label"],
    },
}


class PromptFamily(str, Enum):
    """Supported prompt families.

    Families control how much instruction and negation detail is included.
    """

    LABEL_V1 = "label_v1"
    DESC_V1 = "desc_v1"
    DESC_NEG_V1 = "desc_neg_v1"


PROMPTS_LABEL: Dict[str, str] = {
    "default": "Segment all clinically relevant structures using concise labels. If no target structures are present, return [].",
    "polyp": 'Segment: colorectal polyp. Use the label "colorectal polyp". If no colorectal polyp is present, return [].',
    "derm_lesion": 'Segment: skin lesion region(s). Use the label "skin lesion". If no skin lesion is present, return [].',
    "ima_plusplus": 'Segment: skin lesion region(s). Use the label "skin lesion". If no skin lesion is present, return [].',
    "optic_disc_cup": (
        'Segment the optic disc and the optic cup in this retinal fundus image. '
        'One entry should have the label "optic disc" and the other should have the label "optic cup". '
        "If neither target is present, return []."
    ),
    "laparoscopy_uterus_tools": (
        'In this video frame from a laparoscopic hysterectomy, segment the uterus and any surgical instruments. '
        'Use the label "uterus" for the uterus. Use the label "surgical tool" for any and all surgical instruments, '
        "including visible tips, shafts, and manipulators. If a target is not visible, omit it from the JSON list. "
        "If neither target is visible, return []."
    ),
    "busi_mass": 'Segment: breast mass. Use the label "mass". If no breast mass is present, return [].',
    "lits_liver": 'Segment: liver. Use the label "liver". If the liver is not present, return [].',
    "lits_liver_mass": 'Segment: liver mass. Use the label "liver mass". If no liver mass is present, return [].',
    "pneumothorax_cxr": 'Segment: pneumothorax. Use the label "pneumothorax". If no pneumothorax is present, return [].',
    "histopathology": "Segment tissue regions of diagnostic interest using concise labels such as tumor, stroma, or necrosis. If no diagnostic tissue is present, return [].",
}

PROMPTS_DESC: Dict[str, str] = {
    "default": "Segment all clinically relevant anatomy or findings present in the image using precise medical terminology. If no target structures are present, return [].",
    "polyp": (
        'Image context: This is a colonoscopy frame. Segment: colorectal polyp (an abnormal growth of mucosal tissue protruding into the lumen). '
        'Polyps are often pink, red, or tan and may be round, oval, or lobulated. Include the full visible polyp tissue, including the stalk when present. '
        'Use the label "colorectal polyp". If no colorectal polyp is present, return [].'
    ),
    "derm_lesion": (
        'Image context: This is a dermoscopic image of skin. Segment: skin lesion region(s), an abnormal area distinct from surrounding normal skin. '
        'Lesions may be pigmented or erythematous and typically have a visible border. Use the label "skin lesion". '
        "If no skin lesion is present, return []."
    ),
    "ima_plusplus": (
        'Image context: This is a dermoscopic image of skin. Segment: skin lesion region(s), an abnormal area distinct from surrounding normal skin. '
        'Lesions may be pigmented or erythematous and typically have a visible border. Use the label "skin lesion". '
        "If no skin lesion is present, return []."
    ),
    "optic_disc_cup": (
        'Image context: This is a retinal fundus photograph. Segment the optic disc and optic cup. '
        'The optic disc is the bright, roughly oval optic nerve head where the major retinal vessels converge. '
        'The optic cup is the inner central depression within the optic disc. Return tight boundaries, do not swap labels, '
        'and use the labels "optic disc" and "optic cup". If the targets are not present, return [].'
    ),
    "laparoscopy_uterus_tools": (
        'Image context: This is a video frame from a laparoscopic hysterectomy. Segment the uterus and any surgical instruments. '
        'The uterus is the central pelvic soft-tissue organ in the operative field. Surgical tools are elongated instruments with visible tips, shafts, '
        'and manipulators. Use the labels "uterus" and "surgical tool". If a target is not visible, omit it from the JSON list. '
        "If neither target is visible, return []."
    ),
    "busi_mass": (
        'Image context: This is a breast ultrasound image. Segment: breast mass, a focal lesion region with a visible boundary. '
        'The lesion often appears lower echogenicity than adjacent tissue, but use the visible lesion boundary rather than intensity alone. '
        'Use the label "mass". If no breast mass is present, return [].'
    ),
    "lits_liver": (
        'Image context: This is an axial slice of an abdominal CT. Segment: liver, a large solid organ with a smooth outer contour in the upper abdomen. '
        'Use the label "liver". If the liver is not visible, return [].'
    ),
    "lits_liver_mass": (
        'Image context: This is an axial slice of an abdominal CT. Segment: liver mass, a focal lesion region within the liver parenchyma that differs in attenuation or texture from surrounding liver. '
        'Use the label "liver mass". If no liver mass is present, return [].'
    ),
    "pneumothorax_cxr": (
        'Image context: This is a chest radiograph. Segment: pneumothorax (pleural air). '
        'This typically appears as a peripheral region with absent lung markings beyond a visible pleural line between the collapsed lung and chest wall or mediastinum. '
        'Use the label "pneumothorax". If no pneumothorax is present, return [].'
    ),
    "histopathology": "Image context: This is a histopathology image. Segment tissue regions of diagnostic interest such as tumor, stroma, or necrosis using descriptive labels. If no diagnostic tissue is present, return [].",
}

PROMPTS_NEGATION: Dict[str, str] = {
    "default": "Return an empty list if no target structures are present. Do not include rulers, overlays, text, or other artifacts.",
    "polyp": (
        "Return a tight boundary. Exclude: specular glare/highlights, bubbles/debris, the endoscope tip, any instruments, and any "
        "on-screen text/overlays. Do not include surrounding normal mucosa. Use the label \"colorectal polyp\". If no colorectal polyp "
        "is present, return []."
    ),
    "derm_lesion": (
        "Return a tight boundary around lesion pixels. Exclude: hairs, ruler marks, color calibration charts, ink/pen markings, specular "
        "highlights, and any text/overlays. Use the label \"skin lesion\". If no skin lesion is present, return []."
    ),
    "ima_plusplus": (
        "Return a tight boundary around lesion pixels. Exclude: hairs, ruler marks, color calibration charts, ink/pen markings, specular "
        "highlights, and any text/overlays. Use the label \"skin lesion\". If no skin lesion is present, return []."
    ),
    "optic_disc_cup": (
        "Return tight boundaries and do not swap labels. Exclude: background outside the retina (black borders), and any on-screen "
        "text/overlays. If the targets are not present, return []."
    ),
    "laparoscopy_uterus_tools": (
        "Return tight boundaries. Exclude: smoke/plume regions and specular glare where they are not clearly part of tissue or tool; "
        "exclude any on-screen text/overlays. If a target is not visible, omit it from the JSON list. If neither is visible, return []."
    ),
    "busi_mass": (
        "Return a tight boundary. Exclude: text overlays, depth markers, measurement calipers/boxes, and ultrasound UI elements. Do not "
        "include posterior acoustic artifacts (shadowing/enhancement) unless they are clearly part of the lesion itself. Use the label "
        "\"mass\". If no breast mass is present, return []."
    ),
    "lits_liver": (
        "Return a tight outer boundary around liver parenchyma. Exclude: spleen, stomach/bowel, gallbladder, kidneys, abdominal wall/soft "
        "tissues outside the liver, major vessels outside the liver boundary, and any external annotations/overlays. Use the label "
        "\"liver\". If the liver is not visible, return []."
    ),
    "lits_liver_mass": (
        "Return a tight boundary around lesion pixels only. Exclude: normal liver parenchyma, major vessels/ducts, non-hepatic organs, "
        "and any external annotations/overlays. Use the label \"liver mass\". If no liver mass is present, return []."
    ),
    "pneumothorax_cxr": (
        "Return a tight boundary around pleural air space. Exclude: normal lung parenchyma, mediastinum/heart, ribs/clavicles/scapulae, "
        "skin folds, and any external labels/lines/overlays. Use the label \"pneumothorax\". If no pneumothorax is present, return []."
    ),
    "histopathology": (
        "Do not mark slide artifacts, pen marks, or background whitespace; return an empty list if no diagnostic tissue is found."
    ),
}


def build_prompt(task: str, family: str | PromptFamily) -> str:
    """Assemble a prompt for a task and family.

    Args:
        task: Task identifier (e.g., ``"polyp"`` or ``"optic_disc_cup"``).
        family: Prompt family name or :class:`PromptFamily` value.

    Returns:
        Full prompt text combining the schema preamble and task instructions.

    Raises:
        KeyError: If the task is unknown for the requested family.
        ValueError: If the family is not supported.
    """

    try:
        prompt_family = PromptFamily(family)
    except ValueError as exc:
        valid = ", ".join(p.value for p in PromptFamily)
        raise ValueError(f"Unsupported prompt family '{family}'. Valid options: {valid}") from exc

    task_key = task.lower()

    if prompt_family == PromptFamily.LABEL_V1:
        body = PROMPTS_LABEL[task_key]
    elif prompt_family == PromptFamily.DESC_V1:
        body = PROMPTS_DESC[task_key]
    else:
        desc_body = PROMPTS_DESC[task_key]
        negation = PROMPTS_NEGATION[task_key]
        body = f"{desc_body}\n\n{negation}"

    return f"{SCHEMA_PREAMBLE}\n\n{body}"


@dataclass(frozen=True)
class ProviderPrompt:
    """Container for provider-specific prompt material."""

    prompt: str
    targets: tuple[str, ...] | None = None
    instructions: Mapping[str, str] | None = None


DEFAULT_TARGETS: Dict[str, List[str]] = {
    "polyp": ["colorectal polyp"],
    "derm_lesion": ["skin lesion"],
    "ima_plusplus": ["skin lesion"],
    "optic_disc_cup": ["optic disc", "optic cup"],
    "laparoscopy_uterus_tools": ["uterus", "surgical tool"],
    "busi_mass": ["mass"],
    "lits_liver": ["liver"],
    "lits_liver_mass": ["liver mass"],
    "pneumothorax_cxr": ["pneumothorax"],
    "histopathology": ["diagnostic tissue"],
}

REPLICATE_TASK_DESCRIPTORS: Dict[str, str] = {
    "polyp": "This is a colonoscopy frame. Return a tight mask around only the visible polyp tissue.",
    "derm_lesion": "This is a dermoscopic skin image. Return a tight mask around only the visible lesion region.",
    "ima_plusplus": "This is a dermoscopic skin image. Return a tight mask around only the visible lesion region.",
    "optic_disc_cup": "This is a retinal fundus image. Return a tight mask around only the visible target region and do not swap optic disc and optic cup.",
    "laparoscopy_uterus_tools": "This is a laparoscopic surgical frame. Return a tight mask around only the visible target region.",
    "busi_mass": "This is a breast ultrasound image. Return a tight mask around only the visible lesion boundary.",
    "lits_liver": "This is an axial abdominal CT slice. Return a tight mask around only the visible liver boundary.",
    "lits_liver_mass": "This is an axial abdominal CT slice. Return a tight mask around only the visible liver lesion.",
    "pneumothorax_cxr": "This is a chest radiograph. Return a tight mask around only the visible pleural air space.",
    "histopathology": "This is a histopathology image. Return a tight mask around only the visible diagnostic tissue region.",
}

REPLICATE_TASK_EXCLUSIONS: Dict[str, str] = {
    "polyp": "surrounding normal mucosa, specular glare, bubbles/debris, instruments, and text overlays",
    "derm_lesion": "hairs, ruler marks, color charts, pen marks, specular glare, and text overlays",
    "ima_plusplus": "hairs, ruler marks, color charts, pen marks, specular glare, and text overlays",
    "optic_disc_cup": "background outside the retina and any text overlays",
    "laparoscopy_uterus_tools": "smoke/plume, glare not belonging to tissue/tool, and text overlays",
    "busi_mass": "UI elements, depth markers, calipers, and non-lesion acoustic artifacts",
    "lits_liver": "non-liver organs, abdominal wall soft tissue, and external overlays",
    "lits_liver_mass": "normal liver parenchyma, non-hepatic organs, vessels/ducts, and external overlays",
    "pneumothorax_cxr": "normal lung, mediastinum, bony structures, skin folds, and external overlays",
    "histopathology": "slide artifacts, pen marks, and background whitespace",
}


def _default_targets(task: str, overrides: Sequence[str] | None = None) -> tuple[str, ...]:
    if overrides:
        return tuple(overrides)
    task_key = task.lower()
    if task_key not in DEFAULT_TARGETS:
        raise KeyError(f"Unknown task '{task}'. Available: {', '.join(sorted(DEFAULT_TARGETS))}")
    return tuple(DEFAULT_TARGETS[task_key])


def _build_replicate_instructions(
    task: str,
    family: PromptFamily,
    targets: Sequence[str],
) -> Mapping[str, str]:
    task_key = task.lower()
    descriptor = REPLICATE_TASK_DESCRIPTORS.get(
        task_key,
        "This is a medical image. Return a tight mask around only the visible target region.",
    )
    exclusions = REPLICATE_TASK_EXCLUSIONS.get(
        task_key,
        "non-target anatomy, background, overlays, and imaging artifacts",
    )

    label_instructions = {label: f"Please segment the {label}." for label in targets}
    if family == PromptFamily.LABEL_V1:
        return label_instructions

    desc_instructions = {
        label: f"Please segment the {label}. {descriptor}"
        for label in targets
    }
    if family == PromptFamily.DESC_V1:
        return desc_instructions

    return {
        label: f"{instruction} Exclude: {exclusions}."
        for label, instruction in desc_instructions.items()
    }


def build_prompt_for_provider(
    task: str,
    family: str | PromptFamily,
    provider: str,
    *,
    targets_override: Sequence[str] | None = None,
) -> ProviderPrompt:
    """Render the appropriate prompt payload for a provider.

    Gemini retains the JSON-schema prompt text from :func:`build_prompt`. Other providers
    receive compact label- or instruction-focused text that avoids schema and parsing
    directives.
    """

    normalized_provider = provider.lower()
    targets = _default_targets(task, overrides=targets_override)

    if normalized_provider == "gemini":
        return ProviderPrompt(prompt=build_prompt(task, family))

    if normalized_provider == "moondream":
        primary = targets[0] if targets else task
        return ProviderPrompt(prompt=primary, targets=targets)

    if normalized_provider == "replicate":
        prompt_family = PromptFamily(family)
        instructions = _build_replicate_instructions(task, prompt_family, targets)
        primary_instruction = instructions.get(targets[0], "") if targets else ""
        return ProviderPrompt(
            prompt=primary_instruction,
            targets=targets,
            instructions=instructions,
        )

    raise ValueError(f"Unsupported provider '{provider}'")

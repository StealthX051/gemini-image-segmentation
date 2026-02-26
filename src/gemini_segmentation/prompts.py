"""Prompt construction utilities for segmentation tasks.

This module centralizes prompt fragments so that different prompt families
can be combined consistently across tasks. Each prompt is assembled from a
schema preamble and a task-specific instruction block. The desc_neg family
appends an additional negation block to discourage false positives.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Mapping, Sequence

SCHEMA_PREAMBLE = (
    "Supplementary Methods: Prompt Ablation for Medical Image Segmentation "
    "(Class-name → Description → Description+Negation)\n\n"
    "Overview We evaluated three pre-specified prompt families for each segmentation task: "
    "(A) Class-name prompt: target specified by clinical label only. (B) Description prompt: "
    "adds a short definition plus stable visual descriptors (e.g., rough color/brightness, "
    "shape, location cues) without exclusions. (C) Description+Negation prompt: adds the same "
    "description plus explicit exclusions (negation) for common artifacts and non-target regions.\n\n"
    "All prompts required the model to return a JSON list of segmentation masks. Each JSON entry contained:\n\n"
    '"box_2d": 2D bounding box\n'
    '"mask": segmentation mask\n'
    '"label": text label\n\n'
    "If a target was absent, the model was instructed to return an empty JSON list [] (or omit that entry for multi-target prompts).\n\n"
    "References (prompt-design rationale): [R1] Google Developers Blog: “Conversational image segmentation with Gemini 2.5” "
    "(recommended JSON-output prompt skeleton). [R2] Yuan et al., MICCAI 2025: “TGSAM-2” (prompt-design ablation: class name vs "
    "detailed description).\n\n"
    "Prompt families were pre-specified as an ablation over language specificity while keeping the output schema constant: all "
    "conditions adhered to the vendor-recommended conversational segmentation prompt skeleton and requested structured JSON mask "
    "outputs to minimize formatting variance [1]. Descriptive prompts were formulated as referring expressions that combine a target "
    "label with stable visual/anatomic attributes and simple relations, consistent with the referring image segmentation literature "
    "in which attributes/relations disambiguate instances beyond class-name-only queries [2]. TGSAM-2 reported improved text-guided "
    "medical segmentation when moving from class-name prompts to detailed textual descriptions, motivating our “Description” "
    "condition [3]. Finally, we evaluated a “Description+Negation” condition by adding explicit exclusions for common artifacts and "
    "non-target regions, reflecting constraint-based multi-turn interaction patterns studied in pixel-level reasoning segmentation "
    "and other conversational segmentation settings [1,4].\n\n"
    "References\n"
    "[1] Voigtlaender P, Gabeur V, Doshi R. Conversational image segmentation with Gemini 2.5. Google Developers Blog. July 21, 2025.\n"
    "[2] Ji L, Du Y, Dang Y, Gao W, Zhang H. A survey of methods for addressing the challenges of referring image segmentation. "
    "Neurocomputing. 2024;583:127599. doi:10.1016/j.neucom.2024.127599.\n"
    "[3] Yuan R, Zhou L, Xu J, Li Q, Chen M, Zhang Y, Feng R, Zhang T, Gao S. TGSAM-2: Text-Guided Medical Image Segmentation using "
    "Segment Anything Model 2. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2025. LNCS 15969. "
    "2025:565–574. doi:10.1007/978-3-032-05127-1_54.\n"
    "[4] Cai D, Yang X, Liu Y, Wang D, Feng S, Zhang Y, Poria S. Pixel-Level Reasoning Segmentation via Multi-turn Conversations. "
    "arXiv. 2025. doi:10.48550/arXiv.2502.09447."
)


class PromptFamily(str, Enum):
    """Supported prompt families.

    Families control how much instruction and negation detail is included.
    """

    LABEL_V1 = "label_v1"
    DESC_V1 = "desc_v1"
    DESC_NEG_V1 = "desc_neg_v1"


PROMPTS_LABEL: Dict[str, str] = {
    "default": "Segment all clinically relevant structures using concise labels.",
    "polyp": (
        "Give the segmentation masks for colorectal polyp. Output a JSON list of segmentation masks where each entry contains "
        "the 2D bounding box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\". "
        "Use descriptive labels. Use the label \"colorectal polyp\". If no colorectal polyp is present, return []."
    ),
    "derm_lesion": (
        "Give the segmentation masks for all skin lesions. Output a JSON list of segmentation masks where each entry contains the "
        "2D bounding box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\". "
        "Use the label \"skin lesion\". If no skin lesion is present, return []."
    ),
    "ima_plusplus": (
        "Give the segmentation masks for all skin lesions. Output a JSON list of segmentation masks where each entry contains the "
        "2D bounding box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\". "
        "Use the label \"skin lesion\". If no skin lesion is present, return []."
    ),
    "optic_disc_cup": (
        "Give the segmentation masks for the optic disc and the optic cup in this retinal fundus image. Output a JSON list of two "
        "segmentation masks. One entry should have the label \"optic disc\" and the other \"optic cup\". Each entry must contain "
        "the 2D bounding box in \"box_2d\", the segmentation mask in \"mask\", and the text label in \"label\"."
    ),
    "laparoscopy_uterus_tools": (
        "In this video frame from a laparoscopic hysterectomy, provide segmentation masks for the anatomy and surgical tools. "
        "Output a JSON list of segmentation masks.\n\n"
        "Use the label \"uterus\" for the uterus.\n"
        "Use the label \"surgical tool\" for any and all surgical instruments, including grasping forceps, LigaSure, hooks, and their "
        "shafts and manipulators. Combine all instrument parts into one category. Each JSON entry must contain the 2D bounding box "
        "in the key \"box_2d\", the segmentation mask in the key \"mask\", and the text label in the key \"label\"."
    ),
    "busi_mass": (
        "Give the segmentation mask for the breast mass in this ultrasound image. Output a JSON list of segmentation masks where each "
        "entry contains the 2D bounding box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the "
        "key \"label\". Use a descriptive label like \"mass\". Use the label \"mass\". If no breast mass is present, return []."
    ),
    "lits_liver": (
        "In this slice of an abdominal CT, give the segmentation mask for the liver. Output a JSON list of segmentation masks where each "
        "entry contains the 2D bounding box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the "
        "key \"label\". Use the label \"liver\". If the liver is not present, return []."
    ),
    "lits_liver_mass": (
        "In this slice of an abdominal CT, give the segmentation mask for the liver mass. Output a JSON list of segmentation masks "
        "where each entry contains the 2D bounding box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text "
        "label in the key \"label\". Use the label \"liver mass\". If no liver mass is present, return []."
    ),
    "pneumothorax_cxr": (
        "In this chest x-ray, give the segmentation mask for the pneumothorax. Output a JSON list of segmentation masks where each entry "
        "contains the 2D bounding box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key "
        "\"label\". Use the label \"pneumothorax\". If no pneumothorax is present, return []."
    ),
    "histopathology": "Segment tissue regions of diagnostic interest with concise labels (e.g., tumor, stroma, necrosis).",
}

PROMPTS_DESC: Dict[str, str] = {
    "default": "Segment all clinically relevant anatomy or findings present in the image using precise medical terminology.",
    "polyp": (
        "Give the segmentation masks for the objects. Output a JSON list of segmentation masks where each entry contains the 2D "
        "bounding box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\". Use "
        "descriptive labels. Image context: This is a colonoscopy frame. Segment: colorectal polyp (an abnormal growth of mucosal "
        "tissue protruding into the lumen). Polyps are often pink/red/tan and may be round/oval or lobulated. Include the full visible "
        "polyp tissue (cap and stalk if present). Do not include surrounding normal mucosa. Use the label \"colorectal polyp\". If no "
        "colorectal polyp is present, return []."
    ),
    "derm_lesion": (
        "Give the segmentation masks for the objects. Output a JSON list of segmentation masks where each entry contains the 2D bounding "
        "box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\". Use descriptive "
        "labels. Image context: This is a dermoscopic image of skin. Segment: skin lesion region(s), i.e., an abnormal area distinct "
        "from surrounding normal skin. Lesions may be pigmented (brown/black) or erythematous (red/pink) and typically have a visible "
        "border. Use the label \"skin lesion\". If no skin lesion is present, return []."
    ),
    "ima_plusplus": (
        "Give the segmentation masks for the objects. Output a JSON list of segmentation masks where each entry contains the 2D bounding "
        "box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\". Use descriptive "
        "labels. Image context: This is a dermoscopic image of skin. Segment: skin lesion region(s), i.e., an abnormal area distinct "
        "from surrounding normal skin. Lesions may be pigmented (brown/black) or erythematous (red/pink) and typically have a visible "
        "border. Use the label \"skin lesion\". If no skin lesion is present, return []."
    ),
    "optic_disc_cup": (
        "Give the segmentation masks for the objects. Output a JSON list of two segmentation masks. One entry must have the label "
        "\"optic disc\" and the other \"optic cup\". Each entry must contain the 2D bounding box in the key \"box_2d\", the segmentation "
        "mask in key \"mask\", and the text label in the key \"label\". Image context: This is a retinal fundus photograph. Optic disc: "
        "the bright, roughly oval optic nerve head region where major retinal blood vessels converge. Optic cup: the central inner "
        "depression/inner region within the optic disc. Return tight boundaries and do not swap labels. If the targets are not present, "
        "return []."
    ),
    "laparoscopy_uterus_tools": (
        "Give the segmentation masks for the objects. Output a JSON list of segmentation masks.\n\n"
        "Use the label \"uterus\" for the uterus.\n"
        "Use the label \"surgical tool\" for any and all surgical instruments (tips + shafts + manipulators). Combine all instrument "
        "parts into one category. Each JSON entry must contain the 2D bounding box in the key \"box_2d\", the segmentation mask in key "
        "\"mask\", and the text label in the key \"label\". Use descriptive labels. Image context: This is a video frame from a "
        "laparoscopic hysterectomy. Uterus: the central pelvic soft-tissue organ in the operative field, typically a smooth contiguous "
        "tissue structure. Surgical tools: elongated instruments with visible tips and shafts, often reflective. Return tight boundaries. "
        "If a target is not visible, omit it from the JSON list. If neither is visible, return []."
    ),
    "busi_mass": (
        "Give the segmentation masks for the objects. Output a JSON list of segmentation masks where each entry contains the 2D bounding "
        "box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\". Use descriptive "
        "labels. Image context: This is a breast ultrasound image. Segment: the breast mass / lesion region (a focal lesion region with "
        "a visible boundary). The lesion often appears as lower echogenicity (darker gray) relative to adjacent tissue, but echogenicity "
        "can vary—use the lesion boundary. Use the label \"mass\". If no breast mass is present, return []."
    ),
    "lits_liver": (
        "Give the segmentation masks for the objects. Output a JSON list of segmentation masks where each entry contains the 2D bounding "
        "box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\". Image context: This "
        "is an axial slice of an abdominal CT. Segment: the liver (a large solid organ with a smooth outer contour, occupying a large "
        "portion of the upper abdomen in this slice). Use the label \"liver\". If the liver is not visible, return []."
    ),
    "lits_liver_mass": (
        "Give the segmentation masks for the objects. Output a JSON list of segmentation masks where each entry contains the 2D bounding "
        "box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\". Image context: This "
        "is an axial slice of an abdominal CT. Segment: liver mass (a focal lesion region within the liver parenchyma that differs in "
        "attenuation/texture from surrounding liver in this slice). Use the label \"liver mass\". If no liver mass is present, return []."
    ),
    "pneumothorax_cxr": (
        "Give the segmentation masks for the objects. Output a JSON list of segmentation masks where each entry contains the 2D bounding "
        "box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\". Image context: This is "
        "a chest radiograph. Segment: pneumothorax (pleural air). This typically appears as a peripheral region with absent lung markings "
        "beyond a visible visceral pleural line between collapsed lung and chest wall/mediastinum. Use the label \"pneumothorax\". If no "
        "pneumothorax is present, return []."
    ),
    "histopathology": (
        "Segment tissue regions of diagnostic interest (e.g., tumor, stroma, necrosis) in this histopathology slide using descriptive "
        "labels."
    ),
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

REPLICATE_TASK_CONTEXT: Dict[str, str] = {
    "polyp": "This is a colonoscopy frame.",
    "derm_lesion": "This is a dermoscopic skin image.",
    "ima_plusplus": "This is a dermoscopic skin image.",
    "optic_disc_cup": "This is a retinal fundus image.",
    "laparoscopy_uterus_tools": "This is a laparoscopic surgical frame.",
    "busi_mass": "This is a breast ultrasound image.",
    "lits_liver": "This is an axial abdominal CT slice.",
    "lits_liver_mass": "This is an axial abdominal CT slice.",
    "pneumothorax_cxr": "This is a chest radiograph.",
    "histopathology": "This is a histopathology image.",
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
    context = REPLICATE_TASK_CONTEXT.get(task_key, "This is a medical image.")
    exclusions = REPLICATE_TASK_EXCLUSIONS.get(
        task_key,
        "non-target anatomy, background, overlays, and imaging artifacts",
    )

    label_instructions = {label: f"Segment the {label}." for label in targets}
    if family == PromptFamily.LABEL_V1:
        return label_instructions

    desc_instructions = {
        label: (
            f"{context} Segment the {label}. "
            "Return a tight mask around only the visible target region."
        )
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

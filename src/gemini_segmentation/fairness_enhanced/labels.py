from __future__ import annotations

from dataclasses import dataclass


PROXY_PHRASE = "image-derived perilesional skin tone proxy (ITA)"
BINARY_PHRASE = "lower-ITA (darker-appearing) vs higher-ITA (lighter-appearing) strata"


@dataclass(frozen=True)
class LabelText:
    group_name_short: str
    group_name_long: str
    methods_snippet: str
    figure_caption_snippet: str


def _proxy_phrase(region_strategy: str | None) -> str:
    strategy = (region_strategy or "").strip().lower()
    if strategy == "global_nonlesion":
        return "image-derived non-lesional skin tone proxy (ITA)"
    if strategy == "perilesional_ring":
        return "image-derived perilesional skin tone proxy (ITA)"
    return "image-derived skin tone proxy (ITA)"


def _appearance_phrase(region_strategy: str | None) -> str:
    strategy = (region_strategy or "").strip().lower()
    if strategy == "global_nonlesion":
        return "non-lesional skin"
    if strategy == "perilesional_ring":
        return "perilesional skin"
    return "skin appearance"


def build_label_text(
    *,
    grouping_strategy: str,
    cutoff: float | None = None,
    short_label: str | None = None,
    region_strategy: str | None = None,
) -> LabelText:
    strategy = (grouping_strategy or "binary").strip().lower()
    proxy_phrase = _proxy_phrase(region_strategy)
    appearance = _appearance_phrase(region_strategy)

    if strategy == "binary":
        short = short_label or "Lower ITA / Higher ITA"
        cutoff_txt = f" using a fixed ITA cutoff of {cutoff:.1f}\N{DEGREE SIGN}" if cutoff is not None else ""
        long = (
            "Lower-ITA (darker-appearing "
            + appearance
            + ", image-derived proxy) and Higher-ITA (lighter-appearing "
            + appearance
            + ", image-derived proxy)"
        )
        methods = (
            f"Fairness analyses were stratified by the {proxy_phrase} with {BINARY_PHRASE}{cutoff_txt}."
        )
        caption = (
            f"Groups reflect the {proxy_phrase}; binary strata are reported as {BINARY_PHRASE}{cutoff_txt}."
        )
        return LabelText(short, long, methods, caption)

    if strategy in {"bin6", "6-bin", "6bin"}:
        short = short_label or "ITA 6-bin"
        long = (
            "ITA typology bins (Very Light, Light, Intermediate, Tan, Brown, Dark) from an image-derived "
            + appearance
            + " proxy"
        )
        methods = (
            f"Exploratory analyses used six ITA bins derived from the {proxy_phrase}."
        )
        caption = (
            f"ITA-bin summaries are based on the {proxy_phrase} and should be interpreted as appearance-based proxy strata."
        )
        return LabelText(short, long, methods, caption)

    short = short_label or "Continuous ITA"
    long = "Continuous ITA from image-derived " + appearance + " appearance proxy"
    methods = (
        f"Trend models used continuous values of the {proxy_phrase} without implying patient identity labels."
    )
    caption = (
        f"Curves show model performance vs continuous {proxy_phrase}; these are proxy-based image strata."
    )
    return LabelText(short, long, methods, caption)

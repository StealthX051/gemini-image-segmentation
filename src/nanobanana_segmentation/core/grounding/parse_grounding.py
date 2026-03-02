from __future__ import annotations

from typing import Any, Dict, List


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _normalize_payload(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}

    if hasattr(raw, "model_dump"):
        try:
            dumped = raw.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    if hasattr(raw, "__dict__"):
        out = {}
        for k, v in raw.__dict__.items():
            if not k.startswith("_"):
                out[k] = v
        return out

    return {"value": str(raw)}


def parse_thought_fields(raw_response: Any) -> Dict[str, Any]:
    payload = _normalize_payload(raw_response)
    signatures: List[str] = []
    summaries: List[str] = []

    for node in _walk(payload):
        for key in ("thought_signature", "thoughtSignature", "signature"):
            if key in node and node[key] is not None:
                signatures.append(str(node[key]))

        is_thought = bool(node.get("thought", False)) if isinstance(node, dict) else False
        text = node.get("text") if isinstance(node, dict) else None
        if is_thought and text:
            summaries.append(str(text))

        if "thoughtSummary" in node:
            value = node.get("thoughtSummary")
            if isinstance(value, str) and value.strip():
                summaries.append(value.strip())

    dedup_signatures = list(dict.fromkeys(signatures))
    dedup_summaries = list(dict.fromkeys(summaries))

    return {
        "thought_signature_present": bool(dedup_signatures),
        "thought_signatures": dedup_signatures,
        "thought_summaries": dedup_summaries,
    }


def parse_grounding_fields(raw_response: Any) -> Dict[str, Any]:
    payload = _normalize_payload(raw_response)

    queries: List[Dict[str, str]] = []
    chunks: List[Dict[str, Any]] = []
    supports: List[Dict[str, Any]] = []
    entry_points: List[Dict[str, Any]] = []

    for node in _walk(payload):
        if "webSearchQueries" in node and isinstance(node["webSearchQueries"], list):
            for q in node["webSearchQueries"]:
                queries.append({"type": "text", "query": str(q)})

        if "imageSearchQueries" in node and isinstance(node["imageSearchQueries"], list):
            for q in node["imageSearchQueries"]:
                queries.append({"type": "image", "query": str(q)})

        if "groundingChunks" in node and isinstance(node["groundingChunks"], list):
            for chunk in node["groundingChunks"]:
                if isinstance(chunk, dict):
                    chunks.append(chunk)

        if "groundingSupports" in node and isinstance(node["groundingSupports"], list):
            for support in node["groundingSupports"]:
                if isinstance(support, dict):
                    supports.append(support)

        if "searchEntryPoint" in node and isinstance(node["searchEntryPoint"], dict):
            entry_points.append(node["searchEntryPoint"])

    return {
        "executed_queries": queries,
        "grounding_chunks": chunks,
        "grounding_supports": supports,
        "entry_points": entry_points,
    }

from __future__ import annotations

from nanobanana_segmentation.core.grounding.parse_grounding import (
    parse_grounding_fields,
    parse_thought_fields,
)


def test_parse_grounding_and_thought_fields() -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "summary one", "thought": True, "thoughtSignature": "sig-1"},
                        {"text": "normal text"},
                    ]
                },
                "groundingMetadata": {
                    "webSearchQueries": ["query a"],
                    "imageSearchQueries": ["query b"],
                    "groundingChunks": [{"url": "https://example.com"}],
                    "groundingSupports": [{"id": 1}],
                    "searchEntryPoint": {"provider": "google"},
                },
            }
        ]
    }

    thought = parse_thought_fields(payload)
    grounding = parse_grounding_fields(payload)

    assert thought["thought_signature_present"] is True
    assert "sig-1" in thought["thought_signatures"]
    assert "summary one" in thought["thought_summaries"]

    queries = grounding["executed_queries"]
    assert {q["type"] for q in queries} == {"text", "image"}
    assert len(grounding["grounding_chunks"]) == 1

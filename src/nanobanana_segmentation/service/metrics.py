from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUEST_COUNT = Counter("nanobanana_requests_total", "Total segmentation requests", ["tool_mode"])
QC_FAILURE_COUNT = Counter("nanobanana_qc_failures_total", "QC failures by reason", ["reason"])
SEMANTIC_ATTEMPT_COUNT = Counter("nanobanana_semantic_attempts_total", "Semantic attempts executed")
TRANSPORT_RETRY_COUNT = Counter("nanobanana_transport_retries_total", "Transport retries executed")
LATENCY_HISTOGRAM = Histogram("nanobanana_request_latency_seconds", "Segmentation request latency seconds")


def render_prometheus() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST

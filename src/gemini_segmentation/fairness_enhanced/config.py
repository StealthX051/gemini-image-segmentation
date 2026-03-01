from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


@dataclass
class SourceRootsConfig:
    interop_root: Path = Path("results/interop/isic2017_cli_compat_4074")
    isic2016_root: Path = Path("data/ISIC_2016_Part1")
    isic2017_root: Path = Path("data/ISIC_2017_Part1")
    isic2018_root: Path = Path("data/ISIC_2018_Part1")
    ima_plusplus_root: Path = Path("data/IMAplusplus_cli")


@dataclass
class DedupConfig:
    mode: str = "exact"  # none|exact|near
    near_hamming_threshold: int = 6
    include_near_map: bool = True


@dataclass
class ITAConfig:
    binary_cutoff: float = 28.0
    binary_strategy: str = "fixed"  # fixed|median
    region_strategy: str = "global_nonlesion"  # perilesional_ring|global_nonlesion
    estimator: str = "aggregated_lab"  # aggregated_lab|pixelwise_median
    aggregation_stat: str = "median"  # median|mean|trimmed_mean_sd
    trim_std: float = 1.0
    apply_lstar_window: bool = False
    lstar_window_low_pct: float = 5.0
    lstar_window_high_pct: float = 95.0
    include_legacy_like_sensitivity: bool = False
    ring_outer_frac_min_dim: float = 0.02
    ring_inner_frac_min_dim: float = 0.0
    ring_min_pixels: int = 200
    ring_min_area_frac: float = 0.02
    eps: float = 1e-6
    use_field_mask: bool = True
    field_intensity_floor: int = 5
    very_light_threshold: float = 55.0
    light_threshold: float = 41.0
    intermediate_threshold: float = 28.0
    tan_threshold: float = 10.0
    brown_threshold: float = -30.0


@dataclass
class CovariateConfig:
    enabled: bool = True
    deltae_method: str = "ciede2000"  # ciede2000|deltae76
    valid_pixels_base: str = "field"  # field|ring|lesion_ring
    hair_threshold_quantile: float = 0.95
    specular_lstar_cutoff: float = 92.0
    specular_chroma_cutoff: float = 8.0


@dataclass
class FeaturesConfig:
    profile: str = "balanced"  # balanced|full|minimal
    compute_phash_in_core: bool = False
    hair_mode: str = "lite"  # off|lite|full
    include_specular_in_core: bool = False
    roi_max_dim: int = 768
    hair_max_dim: int = 512
    ring_outer_radius_cap: int = 64
    ring_inner_radius_cap: int = 24
    ring_roi_pad_px: int = 24


@dataclass
class BootstrapConfig:
    n_resamples: int = 5000
    method: str = "bca"  # bca|percentile
    fallback_method: str = "percentile"
    seed: int = 42


@dataclass
class TrendConfig:
    enabled: bool = True
    knots: int = 5
    degree: int = 3
    bootstrap_resamples: int = 200
    quantile: float = 0.5
    quantile_alpha: float = 1e-3


@dataclass
class SensitivityConfig:
    enabled: bool = True
    success_thresholds: List[float] = field(default_factory=lambda: [0.30, 0.40, 0.50, 0.60])
    include_near_dedup: bool = True
    include_dependence: bool = True
    include_mask_source: bool = True


@dataclass
class RuntimeConfig:
    stage: str = "all"  # all|core|sensitivity|augment
    resume: bool = True
    checkpoint_every: int = 50
    max_inflight_tasks: int = 0  # 0 => auto (workers)
    workers_auto: bool = True
    memory_target_frac: float = 0.65
    per_worker_estimate_mb_balanced: int = 1200
    per_worker_estimate_mb_full: int = 1800
    per_worker_estimate_mb_minimal: int = 800


@dataclass
class EnhancedFairnessConfig:
    sources: SourceRootsConfig = field(default_factory=SourceRootsConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    ita: ITAConfig = field(default_factory=ITAConfig)
    covariates: CovariateConfig = field(default_factory=CovariateConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    trends: TrendConfig = field(default_factory=TrendConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    allow_image_id_fallback: bool = False
    duplicate_examples_limit: int = 6
    refresh_source_index: bool = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_raw_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        return json.loads(text)
    return yaml.safe_load(text) or {}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _coerce_source_roots(raw: Dict[str, Any]) -> SourceRootsConfig:
    repo = _repo_root()

    def _resolve(value: Any, default: Path) -> Path:
        token = str(value) if value is not None else str(default)
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = (repo / candidate).resolve()
        return candidate

    defaults = SourceRootsConfig()
    return SourceRootsConfig(
        interop_root=_resolve(raw.get("interop_root"), defaults.interop_root),
        isic2016_root=_resolve(raw.get("isic2016_root"), defaults.isic2016_root),
        isic2017_root=_resolve(raw.get("isic2017_root"), defaults.isic2017_root),
        isic2018_root=_resolve(raw.get("isic2018_root"), defaults.isic2018_root),
        ima_plusplus_root=_resolve(raw.get("ima_plusplus_root"), defaults.ima_plusplus_root),
    )


def _as_float_list(values: Iterable[Any], fallback: List[float]) -> List[float]:
    try:
        return [float(v) for v in values]
    except Exception:
        return fallback


def default_enhanced_config() -> EnhancedFairnessConfig:
    return EnhancedFairnessConfig(
        sources=_coerce_source_roots({}),
    )


def load_enhanced_config(path: Path | None) -> EnhancedFairnessConfig:
    cfg = default_enhanced_config()
    if path is None:
        return cfg

    raw = _load_raw_config(path)

    sources_raw = raw.get("sources") or {}
    dedup_raw = raw.get("dedup") or {}
    ita_raw = raw.get("ita") or {}
    cov_raw = raw.get("covariates") or {}
    features_raw = raw.get("features") or {}
    boot_raw = raw.get("bootstrap") or {}
    trend_raw = raw.get("trends") or {}
    sens_raw = raw.get("sensitivity") or {}
    runtime_raw = raw.get("runtime") or {}

    return EnhancedFairnessConfig(
        sources=_coerce_source_roots(sources_raw),
        dedup=DedupConfig(
            mode=str(dedup_raw.get("mode", cfg.dedup.mode)).strip().lower(),
            near_hamming_threshold=int(
                dedup_raw.get("near_hamming_threshold", cfg.dedup.near_hamming_threshold)
            ),
            include_near_map=bool(dedup_raw.get("include_near_map", cfg.dedup.include_near_map)),
        ),
        ita=ITAConfig(
            binary_cutoff=float(ita_raw.get("binary_cutoff", cfg.ita.binary_cutoff)),
            binary_strategy=str(ita_raw.get("binary_strategy", cfg.ita.binary_strategy)).strip().lower(),
            region_strategy=str(
                ita_raw.get("region_strategy", cfg.ita.region_strategy)
            ).strip().lower(),
            estimator=str(ita_raw.get("estimator", cfg.ita.estimator)).strip().lower(),
            aggregation_stat=str(
                ita_raw.get("aggregation_stat", cfg.ita.aggregation_stat)
            ).strip().lower(),
            trim_std=float(ita_raw.get("trim_std", cfg.ita.trim_std)),
            apply_lstar_window=bool(
                ita_raw.get("apply_lstar_window", cfg.ita.apply_lstar_window)
            ),
            lstar_window_low_pct=float(
                ita_raw.get("lstar_window_low_pct", cfg.ita.lstar_window_low_pct)
            ),
            lstar_window_high_pct=float(
                ita_raw.get("lstar_window_high_pct", cfg.ita.lstar_window_high_pct)
            ),
            include_legacy_like_sensitivity=bool(
                ita_raw.get(
                    "include_legacy_like_sensitivity",
                    cfg.ita.include_legacy_like_sensitivity,
                )
            ),
            ring_outer_frac_min_dim=float(
                ita_raw.get("ring_outer_frac_min_dim", cfg.ita.ring_outer_frac_min_dim)
            ),
            ring_inner_frac_min_dim=float(
                ita_raw.get("ring_inner_frac_min_dim", cfg.ita.ring_inner_frac_min_dim)
            ),
            ring_min_pixels=int(ita_raw.get("ring_min_pixels", cfg.ita.ring_min_pixels)),
            ring_min_area_frac=float(ita_raw.get("ring_min_area_frac", cfg.ita.ring_min_area_frac)),
            eps=float(ita_raw.get("eps", cfg.ita.eps)),
            use_field_mask=bool(ita_raw.get("use_field_mask", cfg.ita.use_field_mask)),
            field_intensity_floor=int(
                ita_raw.get("field_intensity_floor", cfg.ita.field_intensity_floor)
            ),
            very_light_threshold=float(
                ita_raw.get("very_light_threshold", cfg.ita.very_light_threshold)
            ),
            light_threshold=float(ita_raw.get("light_threshold", cfg.ita.light_threshold)),
            intermediate_threshold=float(
                ita_raw.get("intermediate_threshold", cfg.ita.intermediate_threshold)
            ),
            tan_threshold=float(ita_raw.get("tan_threshold", cfg.ita.tan_threshold)),
            brown_threshold=float(ita_raw.get("brown_threshold", cfg.ita.brown_threshold)),
        ),
        covariates=CovariateConfig(
            enabled=bool(cov_raw.get("enabled", cfg.covariates.enabled)),
            deltae_method=str(cov_raw.get("deltae_method", cfg.covariates.deltae_method)).strip().lower(),
            valid_pixels_base=str(
                cov_raw.get("valid_pixels_base", cfg.covariates.valid_pixels_base)
            ).strip().lower(),
            hair_threshold_quantile=float(
                cov_raw.get("hair_threshold_quantile", cfg.covariates.hair_threshold_quantile)
            ),
            specular_lstar_cutoff=float(
                cov_raw.get("specular_lstar_cutoff", cfg.covariates.specular_lstar_cutoff)
            ),
            specular_chroma_cutoff=float(
                cov_raw.get("specular_chroma_cutoff", cfg.covariates.specular_chroma_cutoff)
            ),
        ),
        features=FeaturesConfig(
            profile=str(features_raw.get("profile", cfg.features.profile)).strip().lower(),
            compute_phash_in_core=bool(
                features_raw.get("compute_phash_in_core", cfg.features.compute_phash_in_core)
            ),
            hair_mode=str(features_raw.get("hair_mode", cfg.features.hair_mode)).strip().lower(),
            include_specular_in_core=bool(
                features_raw.get("include_specular_in_core", cfg.features.include_specular_in_core)
            ),
            roi_max_dim=max(64, int(features_raw.get("roi_max_dim", cfg.features.roi_max_dim))),
            hair_max_dim=max(64, int(features_raw.get("hair_max_dim", cfg.features.hair_max_dim))),
            ring_outer_radius_cap=max(
                1,
                int(features_raw.get("ring_outer_radius_cap", cfg.features.ring_outer_radius_cap)),
            ),
            ring_inner_radius_cap=max(
                0,
                int(features_raw.get("ring_inner_radius_cap", cfg.features.ring_inner_radius_cap)),
            ),
            ring_roi_pad_px=max(0, int(features_raw.get("ring_roi_pad_px", cfg.features.ring_roi_pad_px))),
        ),
        bootstrap=BootstrapConfig(
            n_resamples=int(boot_raw.get("n_resamples", cfg.bootstrap.n_resamples)),
            method=str(boot_raw.get("method", cfg.bootstrap.method)).strip().lower(),
            fallback_method=str(
                boot_raw.get("fallback_method", cfg.bootstrap.fallback_method)
            ).strip().lower(),
            seed=int(boot_raw.get("seed", cfg.bootstrap.seed)),
        ),
        trends=TrendConfig(
            enabled=bool(trend_raw.get("enabled", cfg.trends.enabled)),
            knots=int(trend_raw.get("knots", cfg.trends.knots)),
            degree=int(trend_raw.get("degree", cfg.trends.degree)),
            bootstrap_resamples=int(
                trend_raw.get("bootstrap_resamples", cfg.trends.bootstrap_resamples)
            ),
            quantile=float(trend_raw.get("quantile", cfg.trends.quantile)),
            quantile_alpha=float(trend_raw.get("quantile_alpha", cfg.trends.quantile_alpha)),
        ),
        sensitivity=SensitivityConfig(
            enabled=bool(sens_raw.get("enabled", cfg.sensitivity.enabled)),
            success_thresholds=_as_float_list(
                sens_raw.get("success_thresholds", cfg.sensitivity.success_thresholds),
                cfg.sensitivity.success_thresholds,
            ),
            include_near_dedup=bool(
                sens_raw.get("include_near_dedup", cfg.sensitivity.include_near_dedup)
            ),
            include_dependence=bool(
                sens_raw.get("include_dependence", cfg.sensitivity.include_dependence)
            ),
            include_mask_source=bool(
                sens_raw.get("include_mask_source", cfg.sensitivity.include_mask_source)
            ),
        ),
        runtime=RuntimeConfig(
            stage=str(runtime_raw.get("stage", cfg.runtime.stage)).strip().lower(),
            resume=bool(runtime_raw.get("resume", cfg.runtime.resume)),
            checkpoint_every=max(
                1,
                int(runtime_raw.get("checkpoint_every", cfg.runtime.checkpoint_every)),
            ),
            max_inflight_tasks=max(
                0,
                int(runtime_raw.get("max_inflight_tasks", cfg.runtime.max_inflight_tasks)),
            ),
            workers_auto=bool(runtime_raw.get("workers_auto", cfg.runtime.workers_auto)),
            memory_target_frac=float(
                runtime_raw.get("memory_target_frac", cfg.runtime.memory_target_frac)
            ),
            per_worker_estimate_mb_balanced=max(
                128,
                int(
                    runtime_raw.get(
                        "per_worker_estimate_mb_balanced",
                        cfg.runtime.per_worker_estimate_mb_balanced,
                    )
                ),
            ),
            per_worker_estimate_mb_full=max(
                128,
                int(
                    runtime_raw.get(
                        "per_worker_estimate_mb_full",
                        cfg.runtime.per_worker_estimate_mb_full,
                    )
                ),
            ),
            per_worker_estimate_mb_minimal=max(
                128,
                int(
                    runtime_raw.get(
                        "per_worker_estimate_mb_minimal",
                        cfg.runtime.per_worker_estimate_mb_minimal,
                    )
                ),
            ),
        ),
        allow_image_id_fallback=bool(
            raw.get("allow_image_id_fallback", cfg.allow_image_id_fallback)
        ),
        duplicate_examples_limit=int(
            raw.get("duplicate_examples_limit", cfg.duplicate_examples_limit)
        ),
        refresh_source_index=bool(raw.get("refresh_source_index", cfg.refresh_source_index)),
    )


def config_to_dict(cfg: EnhancedFairnessConfig) -> Dict[str, Any]:
    return {
        "sources": {
            "interop_root": str(cfg.sources.interop_root),
            "isic2016_root": str(cfg.sources.isic2016_root),
            "isic2017_root": str(cfg.sources.isic2017_root),
            "isic2018_root": str(cfg.sources.isic2018_root),
            "ima_plusplus_root": str(cfg.sources.ima_plusplus_root),
        },
        "dedup": {
            "mode": cfg.dedup.mode,
            "near_hamming_threshold": cfg.dedup.near_hamming_threshold,
            "include_near_map": cfg.dedup.include_near_map,
        },
        "ita": {
            "binary_cutoff": cfg.ita.binary_cutoff,
            "binary_strategy": cfg.ita.binary_strategy,
            "region_strategy": cfg.ita.region_strategy,
            "estimator": cfg.ita.estimator,
            "aggregation_stat": cfg.ita.aggregation_stat,
            "trim_std": cfg.ita.trim_std,
            "apply_lstar_window": cfg.ita.apply_lstar_window,
            "lstar_window_low_pct": cfg.ita.lstar_window_low_pct,
            "lstar_window_high_pct": cfg.ita.lstar_window_high_pct,
            "include_legacy_like_sensitivity": cfg.ita.include_legacy_like_sensitivity,
            "ring_outer_frac_min_dim": cfg.ita.ring_outer_frac_min_dim,
            "ring_inner_frac_min_dim": cfg.ita.ring_inner_frac_min_dim,
            "ring_min_pixels": cfg.ita.ring_min_pixels,
            "ring_min_area_frac": cfg.ita.ring_min_area_frac,
            "eps": cfg.ita.eps,
            "use_field_mask": cfg.ita.use_field_mask,
            "field_intensity_floor": cfg.ita.field_intensity_floor,
            "very_light_threshold": cfg.ita.very_light_threshold,
            "light_threshold": cfg.ita.light_threshold,
            "intermediate_threshold": cfg.ita.intermediate_threshold,
            "tan_threshold": cfg.ita.tan_threshold,
            "brown_threshold": cfg.ita.brown_threshold,
        },
        "covariates": {
            "enabled": cfg.covariates.enabled,
            "deltae_method": cfg.covariates.deltae_method,
            "valid_pixels_base": cfg.covariates.valid_pixels_base,
            "hair_threshold_quantile": cfg.covariates.hair_threshold_quantile,
            "specular_lstar_cutoff": cfg.covariates.specular_lstar_cutoff,
            "specular_chroma_cutoff": cfg.covariates.specular_chroma_cutoff,
        },
        "features": {
            "profile": cfg.features.profile,
            "compute_phash_in_core": cfg.features.compute_phash_in_core,
            "hair_mode": cfg.features.hair_mode,
            "include_specular_in_core": cfg.features.include_specular_in_core,
            "roi_max_dim": cfg.features.roi_max_dim,
            "hair_max_dim": cfg.features.hair_max_dim,
            "ring_outer_radius_cap": cfg.features.ring_outer_radius_cap,
            "ring_inner_radius_cap": cfg.features.ring_inner_radius_cap,
            "ring_roi_pad_px": cfg.features.ring_roi_pad_px,
        },
        "bootstrap": {
            "n_resamples": cfg.bootstrap.n_resamples,
            "method": cfg.bootstrap.method,
            "fallback_method": cfg.bootstrap.fallback_method,
            "seed": cfg.bootstrap.seed,
        },
        "trends": {
            "enabled": cfg.trends.enabled,
            "knots": cfg.trends.knots,
            "degree": cfg.trends.degree,
            "bootstrap_resamples": cfg.trends.bootstrap_resamples,
            "quantile": cfg.trends.quantile,
            "quantile_alpha": cfg.trends.quantile_alpha,
        },
        "sensitivity": {
            "enabled": cfg.sensitivity.enabled,
            "success_thresholds": list(cfg.sensitivity.success_thresholds),
            "include_near_dedup": cfg.sensitivity.include_near_dedup,
            "include_dependence": cfg.sensitivity.include_dependence,
            "include_mask_source": cfg.sensitivity.include_mask_source,
        },
        "runtime": {
            "stage": cfg.runtime.stage,
            "resume": cfg.runtime.resume,
            "checkpoint_every": cfg.runtime.checkpoint_every,
            "max_inflight_tasks": cfg.runtime.max_inflight_tasks,
            "workers_auto": cfg.runtime.workers_auto,
            "memory_target_frac": cfg.runtime.memory_target_frac,
            "per_worker_estimate_mb_balanced": cfg.runtime.per_worker_estimate_mb_balanced,
            "per_worker_estimate_mb_full": cfg.runtime.per_worker_estimate_mb_full,
            "per_worker_estimate_mb_minimal": cfg.runtime.per_worker_estimate_mb_minimal,
        },
        "allow_image_id_fallback": cfg.allow_image_id_fallback,
        "duplicate_examples_limit": cfg.duplicate_examples_limit,
        "refresh_source_index": cfg.refresh_source_index,
    }

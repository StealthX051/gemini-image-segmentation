# Validation Snapshots

History-only validation notes live here to preserve reproducibility details without bloating day-to-day onboarding docs.
Do not treat this file as the current recommended workflow or current default configuration.

## Replicate Sa2VA Validation Snapshot (2026-02-19)

### Status
- Focused parity/unit tests passed for Replicate-targeted suites.
- Direct 10-image Replicate smoke run succeeded after account funding was enabled.
- Full polyp 3-family Replicate batch run succeeded with run ID:
  - `replicate_sa2va_polyp_full_20260219-162118`

### Validated Model Version
- `bytedance/sa2va-26b-image:addd35cc4f8e0761836ff1e4af324bd7b1f4fa67ee3d384b69202cb288a7dd4f`

### Full-Run Artifact Roots
- `results/polyp/bytedance_sa2va-26b-image_addd35cc4f8e0761836ff1e4af324bd7b1f4fa67ee3d384b69202cb288a7dd4f/label_v1-33499fdf/replicate_sa2va_polyp_full_20260219-162118`
- `results/polyp/bytedance_sa2va-26b-image_addd35cc4f8e0761836ff1e4af324bd7b1f4fa67ee3d384b69202cb288a7dd4f/desc_v1-a60ffa93/replicate_sa2va_polyp_full_20260219-162118`
- `results/polyp/bytedance_sa2va-26b-image_addd35cc4f8e0761836ff1e4af324bd7b1f4fa67ee3d384b69202cb288a7dd4f/desc_neg_v1-b99674e5/replicate_sa2va_polyp_full_20260219-162118`

### Full-Run Summary Highlights
- `label_v1`: mean IoU `0.7147`, mean Dice `0.7909`, success rate `0.775` (1000 predictions).
- `desc_v1`: mean IoU `0.7119`, mean Dice `0.7882`, success rate `0.772` (1000 predictions).
- `desc_neg_v1`: mean IoU `0.7231`, mean Dice `0.7944`, success rate `0.796` (1000 predictions).

## Replicate Smoke Command Variants

### Funded Accounts / Standard Parity
```powershell
$env:POLYP_DATASET_ROOT = "D:\Projects\gemini_image_segmentation\segmented-images"
python -m gemini_segmentation.cli segment polyp "$env:POLYP_DATASET_ROOT" `
  --provider replicate `
  --replicate-model-version bytedance/sa2va-26b-image:addd35cc4f8e0761836ff1e4af324bd7b1f4fa67ee3d384b69202cb288a7dd4f `
  --prompt-family label_v1 `
  --sample-size 10 `
  --workers 10 `
  --rate-limit 0.5 `
  --max-retries 5 `
  --local-cache `
  --local-cache-dir results/.request_cache `
  --replicate-cache-dir results/.replicate_mask_cache `
  --run-id replicate_sa2va_smoke_YYYYMMDD-HHMMSS
```

### Throttled Accounts / Conservative Fallback
```powershell
$env:POLYP_DATASET_ROOT = "D:\Projects\gemini_image_segmentation\segmented-images"
python -m gemini_segmentation.cli segment polyp "$env:POLYP_DATASET_ROOT" `
  --provider replicate `
  --replicate-model-version bytedance/sa2va-26b-image:addd35cc4f8e0761836ff1e4af324bd7b1f4fa67ee3d384b69202cb288a7dd4f `
  --prompt-family label_v1 `
  --sample-size 10 `
  --workers 1 `
  --rate-limit 12 `
  --max-retries 2 `
  --local-cache `
  --local-cache-dir results/.request_cache `
  --replicate-cache-dir results/.replicate_mask_cache `
  --run-id replicate_sa2va_smoke_20260219_slow
```

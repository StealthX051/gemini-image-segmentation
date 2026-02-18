# Legacy Notebooks

The repository includes legacy notebook stages that predate the modular CLI. They remain useful for provenance and dataset-specific experimentation.

## Notebook Families
- `01_*` + `02_*`: polyp preparation and evaluation.
- `03_*` + `04_*`: optic disc/cup preparation and evaluation.
- `05_*` + `06_*`: dermatology preparation and evaluation.
- `07_*` + `08_*`: BUSI preparation and evaluation.
- `09_*` + `10_*`: chest X-ray pneumothorax preparation and evaluation.
- `11_*` + `12_*`: LiTS preparation and evaluation.
- `13_*` + `14_*`: histopathology preparation and evaluation (copy variants).
- `15_*` + `16_*`: laparoscopy preparation and evaluation (copy variants).
- `ita_fitzpatrick_analysis.ipynb`: fairness/ITA analysis from segmentation outputs.
- `2_01_polyp_medoid_selection.ipynb` and `98_vasc_surg_working.ipynb`: ad hoc exploratory stages.

## Relationship To CLI
- CLI modules in `src/gemini_segmentation/` are the canonical, testable implementation of the production workflow.
- Notebook logic should generally be migrated into `src/` before significant extension.
- Use notebooks for exploratory work, debugging, and qualitative review, not as the primary integration surface.

## Editing Guidance
- Avoid mass reformatting or output clearing unless explicitly requested.
- If notebook behavior is promoted into CLI code, add tests under `tests/` and document migration notes in `README.md`.
- Keep dataset path assumptions explicit; do not silently relocate source files.

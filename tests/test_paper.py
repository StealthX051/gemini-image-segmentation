from pathlib import Path

import pandas as pd
import pytest

from gemini_segmentation.paper.config import load_paper_config
from gemini_segmentation.paper.make_all import DEFAULT_CONFIG_PATH, generate_artifacts


def test_load_paper_config_round_trip():
    config = load_paper_config(DEFAULT_CONFIG_PATH)
    assert config.required_columns
    assert config.tables
    assert config.figures


def test_generate_artifacts_creates_outputs(tmp_path: Path):
    results = pd.DataFrame(
        {
            "task": ["polyp", "polyp", "optic"],
            "model": ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-pro"],
            "prompt_strategy": ["label_v1", "label_v1", "desc_v1"],
            "iou": [0.7, 0.8, 0.6],
            "dice": [0.72, 0.83, 0.62],
            "success": [1, 1, 0],
        }
    )
    results_path = tmp_path / "results.csv"
    results.to_csv(results_path, index=False)

    artifacts_dir = tmp_path / "artifacts"
    generate_artifacts(results_path, artifacts_dir=artifacts_dir)

    tables_dir = artifacts_dir / "tables"
    figures_dir = artifacts_dir / "figures"
    assert (tables_dir / "table1.csv").exists()
    assert (tables_dir / "table1.html").exists()
    assert (tables_dir / "table1.docx").exists()
    assert (figures_dir / "figure1.png").exists()
    assert (figures_dir / "figure1.pdf").exists()

    table_df = pd.read_csv(tables_dir / "table1.csv")
    assert "Iou Mean" in table_df.columns
    assert not table_df.empty


def test_generate_artifacts_parquet(tmp_path: Path):
    results = pd.DataFrame(
        {
            "task": ["polyp", "optic"],
            "model": ["gemini-2.5-flash", "gemini-1.5-pro"],
            "prompt_strategy": ["label_v1", "desc_v1"],
            "iou": [0.75, 0.6],
            "dice": [0.8, 0.62],
            "success": [1, 0],
        }
    )
    results_path = tmp_path / "results.parquet"
    results.to_parquet(results_path)

    artifacts_dir = tmp_path / "artifacts"
    generate_artifacts(results_path, artifacts_dir=artifacts_dir)

    assert (artifacts_dir / "tables" / "table1.csv").exists()


def test_generate_artifacts_missing_columns(tmp_path: Path):
    results = pd.DataFrame(
        {
            "task": ["polyp"],
            "model": ["gemini-2.5-flash"],
            "prompt_strategy": ["label_v1"],
            "iou": [0.7],
            "success": [1],
        }
    )
    results_path = tmp_path / "results.csv"
    results.to_csv(results_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns: dice"):
        generate_artifacts(results_path, artifacts_dir=tmp_path / "artifacts")

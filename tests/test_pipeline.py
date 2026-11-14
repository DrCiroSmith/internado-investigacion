"""
Smoke and validation tests for the Clinical Research & Epidemiological Time-Series Platform.
"""
import csv
from pathlib import Path
import sys
import tempfile

# Add repo root to sys.path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_synthetic_clinical_data import generate_synthetic_timeseries, export_to_csv


def test_generate_synthetic_timeseries_structure():
    records = generate_synthetic_timeseries(start_date_str="2024-01-01", num_weeks=52, seed=123)
    assert len(records) == 52, "Should generate exactly 52 weekly records"
    
    first = records[0]
    expected_keys = {
        "encounter_week", "year", "iso_week",
        "surgical_exodontia", "restorative_procedures",
        "preventive_prophylaxis", "emergency_triage",
        "total_encounters", "exodontia_ratio"
    }
    assert expected_keys.issubset(set(first.keys()))


def test_generate_synthetic_timeseries_metric_integrity():
    records = generate_synthetic_timeseries(num_weeks=26, seed=999)
    for row in records:
        assert row["surgical_exodontia"] > 0
        assert row["restorative_procedures"] > 0
        assert row["preventive_prophylaxis"] > 0
        assert row["emergency_triage"] > 0
        assert row["total_encounters"] == (
            row["surgical_exodontia"] +
            row["restorative_procedures"] +
            row["preventive_prophylaxis"] +
            row["emergency_triage"]
        )
        assert 0.0 < row["exodontia_ratio"] < 1.0


def test_reproducibility():
    run_a = generate_synthetic_timeseries(num_weeks=20, seed=42)
    run_b = generate_synthetic_timeseries(num_weeks=20, seed=42)
    assert run_a == run_b, "Generator must be deterministic given identical seed"


def test_csv_export_roundtrip():
    records = generate_synthetic_timeseries(num_weeks=10, seed=7)
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test_export.csv"
        export_to_csv(records, str(csv_file))
        
        assert csv_file.exists()
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            loaded = list(reader)
            
        assert len(loaded) == 10
        assert loaded[0]["encounter_week"] == records[0]["encounter_week"]
        assert int(loaded[0]["surgical_exodontia"]) == records[0]["surgical_exodontia"]


def test_docker_compose_configuration():
    compose_path = REPO_ROOT / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml must exist at repo root"
    content = compose_path.read_text(encoding="utf-8")
    assert "jupyter:" in content
    assert "mysql:" in content
    assert "8888:8888" in content
    assert "MYSQL_DATABASE" in content

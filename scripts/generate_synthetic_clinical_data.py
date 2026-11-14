#!/usr/bin/env python3
"""
Synthetic Clinical Dataset Generator for Epidemiological Time-Series Analysis.
Generates fully anonymized, realistic longitudinal dental and clinical procedure logs
with configurable seasonal variations, baseline trends, and holiday dips.

Zero real patient identifiers; 100% synthetic mathematical modeling.
"""
import argparse
import csv
from datetime import datetime, timedelta
import math
import random
from pathlib import Path


def generate_synthetic_timeseries(
    start_date_str: str = "2024-01-01",
    num_weeks: int = 104,
    seed: int = 42
) -> list:
    """
    Generate synthetic longitudinal clinical encounter records week-by-week.
    
    Returns:
        List of dictionaries containing epidemiological and procedural volume metrics.
    """
    random.seed(seed)
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    
    records = []
    
    # Baseline weekly volumes
    base_exodontia = 45.0
    base_restorative = 110.0
    base_preventive = 85.0
    base_emergency = 22.0
    
    for w in range(num_weeks):
        current_date = start_date + timedelta(weeks=w)
        week_num = current_date.isocalendar()[1]
        
        # Annual sinusoidal seasonal effect (peak in summer/late autumn, dip in Dec/Jan holidays)
        seasonality = math.sin((week_num / 52.0) * 2 * math.pi)
        
        # Year-over-year mild demographic growth (2% annual drift)
        growth_drift = 1.0 + (0.02 * (w / 52.0))
        
        # Holiday dip around Christmas / New Year (weeks 51, 52, 1)
        holiday_factor = 0.55 if week_num in (51, 52, 1) else 1.0
        
        # Gaussian noise
        noise_exodontia = random.gauss(0, 4.0)
        noise_restorative = random.gauss(0, 8.0)
        noise_preventive = random.gauss(0, 6.0)
        noise_emergency = random.gauss(0, 3.0)
        
        exodontia = max(5, int((base_exodontia + (seasonality * 8.0) + noise_exodontia) * growth_drift * holiday_factor))
        restorative = max(10, int((base_restorative + (seasonality * 15.0) + noise_restorative) * growth_drift * holiday_factor))
        preventive = max(10, int((base_preventive + (seasonality * 12.0) + noise_preventive) * growth_drift * holiday_factor))
        emergency = max(2, int((base_emergency + (seasonality * 3.0) + noise_emergency) * growth_drift))
        
        total_encounters = exodontia + restorative + preventive + emergency
        
        record = {
            "encounter_week": current_date.strftime("%Y-%m-%d"),
            "year": current_date.year,
            "iso_week": week_num,
            "surgical_exodontia": exodontia,
            "restorative_procedures": restorative,
            "preventive_prophylaxis": preventive,
            "emergency_triage": emergency,
            "total_encounters": total_encounters,
            "exodontia_ratio": round(exodontia / total_encounters, 4) if total_encounters else 0.0
        }
        records.append(record)
        
    return records


def export_to_csv(records: list, output_path: str):
    """Write records to CSV format."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if not records:
        return
        
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description="Synthetic Clinical Epidemiological Data Generator")
    parser.add_argument("--weeks", type=int, default=104, help="Number of weekly time points (default: 104)")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="data/synthetic_clinical_timeseries.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    records = generate_synthetic_timeseries(args.start, args.weeks, args.seed)
    export_to_csv(records, args.output)
    print(f"✅ Successfully generated {len(records)} synthetic clinical weekly records -> {args.output}")


if __name__ == "__main__":
    main()

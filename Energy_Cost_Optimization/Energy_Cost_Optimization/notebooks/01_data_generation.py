"""
Phase 1: Data Generation
PT Surya Textile Indonesia - Energy Optimization Project
Generates 12-month simulation data for 30 machines at 15-minute intervals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ─── Seed for reproducibility ─────────────────────────────────────────────────
np.random.seed(42)

# ─── Configuration ────────────────────────────────────────────────────────────
START_DATE  = datetime(2025, 1, 1, 0, 0, 0)
END_DATE    = datetime(2025, 12, 31, 23, 45, 0)
INTERVAL    = 15  # minutes

PEAK_START  = 17  # 17:00
PEAK_END    = 22  # 22:00
TARIFF_PEAK   = 1700  # IDR/kWh
TARIFF_NORMAL = 1450  # IDR/kWh

# Machine definition
MACHINES = (
    [(f"MOTOR_{i:02d}",    "Motor")      for i in range(1, 16)] +
    [(f"PUMP_{i:02d}",     "Pump")       for i in range(1, 11)] +
    [(f"COMPRESSOR_{i:02d}", "Compressor") for i in range(1, 6)]
)

# Current range per type (A)
CURRENT_RANGE = {
    "Motor":      (15, 40),
    "Pump":       (10, 25),
    "Compressor": (20, 60),
}

print(f"Total machines    : {len(MACHINES)}")
print(f"Start date        : {START_DATE}")
print(f"End date          : {END_DATE}")

# ─── Timestamp generation ─────────────────────────────────────────────────────
timestamps = pd.date_range(start=START_DATE, end=END_DATE, freq="15min")
print(f"Total timestamps  : {len(timestamps):,}")
print(f"Total records     : {len(timestamps) * len(MACHINES):,}")

# ─── Dataset generation ───────────────────────────────────────────────────────
records = []

for ts in timestamps:
    hour = ts.hour
    is_peak = PEAK_START <= hour < PEAK_END

    for machine_id, machine_type in MACHINES:
        # Voltage: 375–390 V
        voltage = np.random.uniform(375, 390)

        # Current based on machine type
        i_min, i_max = CURRENT_RANGE[machine_type]

        # Add load variation: night (00–06) lower load
        if 0 <= hour < 6:
            load_factor = np.random.uniform(0.55, 0.80)
        elif 6 <= hour < 8:
            load_factor = np.random.uniform(0.80, 1.00)
        elif 8 <= hour < 12:
            load_factor = np.random.uniform(0.90, 1.00)
        elif 12 <= hour < 14:
            load_factor = np.random.uniform(0.70, 0.90)  # lunch dip
        elif 14 <= hour < 18:
            load_factor = np.random.uniform(0.90, 1.00)
        elif 18 <= hour < 22:
            load_factor = np.random.uniform(0.85, 1.00)  # peak hours
        else:
            load_factor = np.random.uniform(0.60, 0.80)

        current = np.random.uniform(i_min, i_max) * load_factor

        # Power factor: 10% of records get bad PF (0.60–0.75)
        if np.random.rand() < 0.10:
            pf = np.random.uniform(0.60, 0.75)
        else:
            pf = np.random.uniform(0.80, 0.95)

        # Electrical calculations
        sqrt3 = 1.7320508
        apparent_power_kva = (sqrt3 * voltage * current) / 1000          # S = √3·V·I / 1000
        active_power_kw    = apparent_power_kva * pf                     # P = S × PF
        reactive_power_kvar = (apparent_power_kva**2 - active_power_kw**2) ** 0.5  # Q = √(S²−P²)

        # Energy for 15-minute interval
        energy_kwh = active_power_kw * 0.25                              # E = P × 0.25

        # Tariff
        tariff = TARIFF_PEAK if is_peak else TARIFF_NORMAL

        records.append({
            "timestamp":          ts,
            "machine_id":         machine_id,
            "machine_type":       machine_type,
            "voltage_v":          round(voltage, 2),
            "current_a":          round(current, 3),
            "power_factor":       round(pf, 4),
            "active_power_kw":    round(active_power_kw, 4),
            "reactive_power_kvar": round(reactive_power_kvar, 4),
            "apparent_power_kva": round(apparent_power_kva, 4),
            "energy_kwh":         round(energy_kwh, 6),
            "tariff_idr_kwh":     tariff,
        })

print("Building DataFrame …")
df = pd.DataFrame(records)

# ─── Save raw data ────────────────────────────────────────────────────────────
out_path = "/home/claude/Energy_Cost_Optimization/data/raw/energy_consumption.csv"
df.to_csv(out_path, index=False)

print(f"\n✅ Dataset saved → {out_path}")
print(f"   Shape          : {df.shape}")
print(f"   Memory usage   : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"\nSample (first 5 rows):")
print(df.head())
print(f"\nBasic stats:")
print(df.describe())

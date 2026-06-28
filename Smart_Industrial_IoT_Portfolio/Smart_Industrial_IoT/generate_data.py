#!/usr/bin/env python3
"""
Smart Industrial IoT — Data Generator
PT Nusantara Steel Manufacturing (Simulation)

Menghasilkan 1.900.000+ record sensor realistis untuk 36 aset industri.
Data mengikuti 7 aturan fisika/engineering yang saling berkorelasi.

Jalankan: python generate_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os, sys, warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER   = os.path.join(BASE_DIR, "data", "master")
RAW      = os.path.join(BASE_DIR, "data", "raw")
PROC     = os.path.join(BASE_DIR, "data", "processed")
for d in [MASTER, RAW, PROC]:
    os.makedirs(d, exist_ok=True)

START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2024, 12, 31, 23, 50)
INTERVAL   = "10min"
CURRENT_YEAR = 2024

TIMESTAMPS = pd.date_range(START_DATE, END_DATE, freq=INTERVAL)
N_STEPS = len(TIMESTAMPS)

# ─────────────────────────────────────────────────────────
# DEFINISI MESIN (36 unit)
# ─────────────────────────────────────────────────────────
# Kolom: machine_id, machine_type, location, rated_power_kw, install_year,
#        current_min, current_max, temp_base, vib_base,
#        has_flow, flow_min, flow_max,
#        has_pressure, press_min, press_max,
#        hum_base, pf_min, pf_max
MACHINES = [
    # ── 15 Motor Induksi ──────────────────────────────────────────────────────────
    ("MTR-001","Motor","Production Hall A",22.0, 2010, 10,32, 50,1.50, False,0,0, False,0,0,   52,0.82,0.94),
    ("MTR-002","Motor","Production Hall A",37.0, 2012, 15,42, 52,1.60, False,0,0, False,0,0,   50,0.83,0.95),
    ("MTR-003","Motor","Production Hall B",55.0, 2009, 20,44, 55,1.80, False,0,0, False,0,0,   53,0.82,0.94),
    ("MTR-004","Motor","Production Hall B",75.0, 2008, 25,45, 58,2.00, False,0,0, False,0,0,   55,0.82,0.93),
    ("MTR-005","Motor","Production Hall C",30.0, 2015, 12,38, 48,1.40, False,0,0, False,0,0,   50,0.83,0.95),
    ("MTR-006","Motor","Production Hall A",15.0, 2018,  8,28, 45,1.20, False,0,0, False,0,0,   48,0.84,0.96),
    ("MTR-007","Motor","Production Hall B",45.0, 2011, 18,43, 53,1.70, False,0,0, False,0,0,   52,0.82,0.94),
    ("MTR-008","Motor","Production Hall C",11.0, 2020,  6,22, 42,1.10, False,0,0, False,0,0,   47,0.85,0.96),
    ("MTR-009","Motor","Production Hall A", 7.5, 2016,  4,18, 40,1.00, False,0,0, False,0,0,   48,0.84,0.96),
    ("MTR-010","Motor","Production Hall B",18.5, 2013,  9,30, 49,1.50, False,0,0, False,0,0,   51,0.83,0.95),
    ("MTR-011","Motor","Production Hall C",22.0, 2017, 10,32, 50,1.30, False,0,0, False,0,0,   49,0.84,0.95),
    ("MTR-012","Motor","Production Hall A",30.0, 2021, 12,38, 47,1.10, False,0,0, False,0,0,   48,0.85,0.96),
    ("MTR-013","Motor","Production Hall B",37.0, 2014, 15,42, 52,1.60, False,0,0, False,0,0,   51,0.83,0.94),
    ("MTR-014","Motor","Production Hall C",55.0, 2019, 20,44, 55,1.40, False,0,0, False,0,0,   50,0.84,0.95),
    ("MTR-015","Motor","Production Hall A",75.0, 2022, 25,45, 56,1.20, False,0,0, False,0,0,   49,0.85,0.96),

    # ── 8 Pompa Industri ──────────────────────────────────────────────────────────
    ("PMP-001","Pump","Pump Station 1", 3.7, 2013,  4,16, 40,1.10, True, 30,90,  True, 2,6,   55,0.83,0.94),
    ("PMP-002","Pump","Pump Station 1", 5.5, 2015,  5,20, 42,1.20, True, 35,100, True, 3,7,   54,0.83,0.94),
    ("PMP-003","Pump","Pump Station 2", 7.5, 2011,  6,24, 44,1.30, True, 40,110, True, 3,7,   56,0.82,0.93),
    ("PMP-004","Pump","Pump Station 1",11.0, 2018,  8,28, 43,1.10, True, 45,120, True, 4,8,   54,0.84,0.94),
    ("PMP-005","Pump","Pump Station 2",15.0, 2014, 10,30, 45,1.20, True, 50,130, True, 4,8,   55,0.83,0.94),
    ("PMP-006","Pump","Pump Station 1",18.5, 2020, 12,30, 44,1.00, True, 55,135, True, 5,9,   53,0.85,0.95),
    ("PMP-007","Pump","Pump Station 2",22.0, 2016, 14,30, 46,1.30, True, 60,140, True, 5,9,   55,0.83,0.94),
    ("PMP-008","Pump","Pump Station 1",11.0, 2022,  8,28, 43,1.00, True, 45,120, True, 4,8,   53,0.85,0.95),

    # ── 6 Air Compressor ─────────────────────────────────────────────────────────
    ("CMP-001","Compressor","Compressor Room",11.0, 2009, 20,40, 58,1.80, True,15,55, True, 5,11,  45,0.82,0.93),
    ("CMP-002","Compressor","Compressor Room",18.5, 2012, 30,55, 60,2.00, True,20,65, True, 6,12,  44,0.82,0.93),
    ("CMP-003","Compressor","Compressor Room",22.0, 2015, 35,62, 61,1.90, True,25,70, True, 6,12,  45,0.83,0.94),
    ("CMP-004","Compressor","Compressor Room",30.0, 2018, 40,68, 63,1.70, True,30,75, True, 7,12,  44,0.83,0.94),
    ("CMP-005","Compressor","Compressor Room",37.0, 2011, 45,70, 62,2.10, True,25,75, True, 7,12,  45,0.82,0.93),
    ("CMP-006","Compressor","Compressor Room",55.0, 2020, 50,70, 65,1.60, True,35,80, True, 8,12,  43,0.84,0.94),

    # ── 4 Conveyor ────────────────────────────────────────────────────────────────
    ("CNV-001","Conveyor","Assembly Line 1", 7.5, 2014,  6,20, 45,2.00, False,0,0, False,0,0,   50,0.82,0.93),
    ("CNV-002","Conveyor","Assembly Line 1",11.0, 2016,  8,24, 47,2.20, False,0,0, False,0,0,   51,0.82,0.93),
    ("CNV-003","Conveyor","Assembly Line 2",15.0, 2015, 10,25, 46,2.10, False,0,0, False,0,0,   50,0.82,0.93),
    ("CNV-004","Conveyor","Assembly Line 2",22.0, 2019, 14,25, 48,1.90, False,0,0, False,0,0,   49,0.83,0.94),

    # ── 2 Boiler (suhu JAUH lebih tinggi) ────────────────────────────────────────
    ("BLR-001","Boiler","Boiler Room",185.0, 2012, 30,75, 100,0.70, True,80,155, True, 6,12,  38,0.86,0.97),
    ("BLR-002","Boiler","Boiler Room",250.0, 2015, 40,80, 105,0.60, True,90,160, True, 7,12,  36,0.87,0.97),

    # ── 1 Cooling Tower (flow TERBESAR) ──────────────────────────────────────────
    ("CLT-001","Cooling Tower","Cooling Area",55.0, 2013, 15,38, 30,0.90, True,150,220, True,1,4,  72,0.84,0.95),
]

MACHINE_COLS = [
    "machine_id","machine_type","location","rated_power_kw","install_year",
    "current_min","current_max","temp_base","vib_base",
    "has_flow","flow_min","flow_max",
    "has_pressure","press_min","press_max",
    "hum_base","pf_min","pf_max",
]

# ─────────────────────────────────────────────────────────
# FUNGSI PEMBANTU
# ─────────────────────────────────────────────────────────
def load_factor(timestamps):
    """Pola beban berdasarkan jam kerja dan hari."""
    h = timestamps.hour.values
    d = timestamps.dayofweek.values
    lf = np.ones(len(timestamps))
    lf = np.where(h < 6,    0.50 + 0.12 * np.random.random(len(lf)), lf)
    lf = np.where((h>=6)&(h<7),  0.70 + 0.10 * np.random.random(len(lf)), lf)
    lf = np.where((h>=7)&(h<18), 0.85 + 0.15 * np.random.random(len(lf)), lf)
    lf = np.where((h>=18)&(h<22),0.72 + 0.10 * np.random.random(len(lf)), lf)
    lf = np.where(h >= 22,  0.55 + 0.10 * np.random.random(len(lf)), lf)
    lf = np.where(d >= 5, lf * 0.50, lf)   # Weekend
    return np.clip(lf, 0.3, 1.1)


def inject_faults(arr, fault_masks, spike_factor, noise_std):
    """Suntikkan anomali ke data selama periode fault."""
    result = arr.copy()
    for mask in fault_masks:
        result[mask] *= spike_factor
        result[mask] += np.random.normal(0, noise_std, mask.sum())
    return result


def build_fault_schedule(m, n_steps, rng):
    """Buat jadwal fault acak — mesin lebih tua → lebih sering fault."""
    age = CURRENT_YEAR - m["install_year"]
    n_faults = max(2, int(age * 0.35) + rng.randint(0, 4))
    masks = []
    fault_events = []
    for _ in range(n_faults):
        start = rng.randint(0, n_steps - 200)
        dur   = rng.randint(6, 36)          # 1–6 jam
        end   = min(start + dur, n_steps)
        mask  = np.zeros(n_steps, dtype=bool)
        mask[start:end] = True
        masks.append(mask)
        fault_events.append({
            "machine_id":   m["machine_id"],
            "start_idx":    start,
            "end_idx":      end,
        })
    return masks, fault_events


# ─────────────────────────────────────────────────────────
# STEP 1 — MACHINE MASTER
# ─────────────────────────────────────────────────────────
def gen_machine_master():
    print("\n[1/6] Machine Master...")
    df = pd.DataFrame(MACHINES, columns=MACHINE_COLS)
    # Pilih kolom yang relevan untuk master
    master = df[["machine_id","machine_type","location","rated_power_kw","install_year"]].copy()
    master.to_csv(os.path.join(MASTER, "machine_master.csv"), index=False)
    print(f"      ✓ {len(master)} mesin disimpan.")
    return df  # Kembalikan full config


# ─────────────────────────────────────────────────────────
# STEP 2 — SENSOR LOG (inti proyek)
# ─────────────────────────────────────────────────────────
def gen_sensor_log(machines_df):
    print("\n[2/6] Sensor Log (ini butuh ~2-3 menit)...")
    rng = np.random.default_rng(SEED)
    LF  = load_factor(TIMESTAMPS)

    all_chunks = []
    all_faults = []

    for idx, row in machines_df.iterrows():
        m = row.to_dict()
        n = N_STEPS

        # ── Load factor per mesin (sedikit variasi) ──
        lf = LF * (0.90 + 0.10 * rng.random(n))

        # ── Age degradation (ATURAN 2) ──
        age = CURRENT_YEAR - m["install_year"]
        age_f = np.clip(age / 25.0, 0, 1)

        # ── Fault schedule ──
        fault_masks, faults = build_fault_schedule(m, n, np.random)
        all_faults.extend(faults)

        # ── Voltage ──
        voltage = 380 + rng.normal(0, 2.0, n)
        voltage = np.clip(voltage, 360, 400)

        # ── Current (ATURAN 4) ──
        i_range = m["current_max"] - m["current_min"]
        current = m["current_min"] + i_range * lf + rng.normal(0, i_range * 0.03, n)
        current = inject_faults(current, fault_masks, spike_factor=1.30, noise_std=i_range*0.05)
        current = np.clip(current, m["current_min"] * 0.7, m["current_max"] * 1.4)

        # ── Temperature (ATURAN 1: current → temp) ──
        curr_norm = (current - m["current_min"]) / (i_range + 1e-6)
        temp = m["temp_base"] + 22 * curr_norm + age_f * 10 + rng.normal(0, 3.0, n)
        temp = inject_faults(temp, fault_masks, spike_factor=1.25, noise_std=4.0)
        # Boiler tetap jauh lebih panas (ATURAN 6)
        temp = np.clip(temp, 20, 180)

        # ── Vibration (ATURAN 2 + 3) ──
        vibration = m["vib_base"] + age_f * 2.0 + 0.5 * curr_norm + rng.normal(0, 0.18, n)
        vibration = inject_faults(vibration, fault_masks, spike_factor=2.20, noise_std=0.3)
        vibration = np.clip(vibration, 0.05, 12.0)

        # ── Humidity (ATURAN 5) ──
        months = TIMESTAMPS.month.values
        seasonal = np.where((months >= 5) & (months <= 10), 8, 0)  # Musim hujan
        humidity = m["hum_base"] + seasonal + rng.normal(0, 4.0, n)
        humidity = np.clip(humidity, 20, 98)

        # ── Frequency ──
        frequency = 50.0 + rng.normal(0, 0.08, n)
        frequency = np.clip(frequency, 49.4, 50.6)

        # ── Power Factor ──
        pf = m["pf_min"] + (m["pf_max"] - m["pf_min"]) * (0.4 + 0.6 * lf) + rng.normal(0, 0.012, n)
        pf = np.clip(pf, 0.65, 1.00)

        # ── Pressure ──
        if m["has_pressure"]:
            pr_range = m["press_max"] - m["press_min"]
            pressure = m["press_min"] + pr_range * lf + rng.normal(0, pr_range * 0.04, n)
            pressure = inject_faults(pressure, fault_masks, spike_factor=1.20, noise_std=pr_range*0.05)
            pressure = np.clip(pressure, 0, m["press_max"] * 1.3)
        else:
            pressure = np.zeros(n)

        # ── Flow Rate (ATURAN 7: Cooling Tower flow terbesar) ──
        if m["has_flow"]:
            fl_range = m["flow_max"] - m["flow_min"]
            flow = m["flow_min"] + fl_range * lf + rng.normal(0, fl_range * 0.04, n)
            flow = np.clip(flow, m["flow_min"] * 0.7, m["flow_max"] * 1.15)
        else:
            flow = np.zeros(n)

        # ── Power kW (ATURAN 4: I → Power → Energy) ──
        power_kw = np.sqrt(3) * voltage * current * pf / 1000

        # ── Energy kWh (per interval 10 menit) ──
        energy_kwh = power_kw * (10 / 60)

        chunk = pd.DataFrame({
            "timestamp":    TIMESTAMPS,
            "machine_id":   m["machine_id"],
            "voltage":      np.round(voltage, 1),
            "current":      np.round(current, 2),
            "temperature":  np.round(temp, 1),
            "vibration":    np.round(vibration, 3),
            "humidity":     np.round(humidity, 1),
            "pressure":     np.round(pressure, 2),
            "flow_rate":    np.round(flow, 1),
            "frequency":    np.round(frequency, 2),
            "power_factor": np.round(pf, 3),
            "power_kw":     np.round(power_kw, 2),
            "energy_kwh":   np.round(energy_kwh, 4),
        })
        all_chunks.append(chunk)

        if (idx + 1) % 6 == 0 or idx == 35:
            print(f"      ✓ {idx+1}/36 mesin diproses...")

    print("      Menggabungkan dan menyimpan (bisa makan waktu)...")
    log = pd.concat(all_chunks, ignore_index=True)
    log.to_csv(os.path.join(RAW, "sensor_log.csv"), index=False)
    print(f"      ✓ {len(log):,} record sensor disimpan → data/raw/sensor_log.csv")
    return log, all_faults


# ─────────────────────────────────────────────────────────
# STEP 3 — ALARM HISTORY (derived dari sensor log)
# ─────────────────────────────────────────────────────────
ALARM_RULES = {
    "temperature":   {"warning": 70, "critical": 80,  "unit": "°C",    "label": "High Temperature"},
    "vibration":     {"warning": 3.5,"critical": 4.5,  "unit": "mm/s",  "label": "High Vibration"},
    "voltage":       {"warning": 365,"critical": 358,  "unit": "V",     "label": "Under Voltage",  "direction": "low"},
    "current":       {"warning": None,"critical": None},  # per-machine
    "power_factor":  {"warning": 0.83,"critical": 0.78, "unit": "",     "label": "Low Power Factor","direction": "low"},
    "pressure":      {"warning": 11.5,"critical": 12.5, "unit": "Bar",  "label": "Over Pressure"},
}

def gen_alarm_history(sensor_log, machines_df):
    print("\n[3/6] Alarm History...")
    alarms = []
    alarm_id = 1

    # Sample 20% of data untuk efisiensi
    sample = sensor_log.sample(frac=0.20, random_state=42).copy()

    # High Temperature alarms
    mask = sample["temperature"] > 70
    df_a = sample[mask].copy()
    df_a["alarm_type"] = "High Temperature"
    df_a["severity"]   = np.where(df_a["temperature"] > 80, "Critical", "Warning")
    df_a["value"]      = df_a["temperature"].astype(str) + " °C"
    alarms.append(df_a[["timestamp","machine_id","alarm_type","severity","value"]])

    # High Vibration alarms (ATURAN 3)
    mask = sample["vibration"] > 3.5
    df_a = sample[mask].copy()
    df_a["alarm_type"] = "High Vibration"
    df_a["severity"]   = np.where(df_a["vibration"] > 4.5, "Critical", "Warning")
    df_a["value"]      = df_a["vibration"].astype(str) + " mm/s"
    alarms.append(df_a[["timestamp","machine_id","alarm_type","severity","value"]])

    # Under Voltage
    mask = sample["voltage"] < 368
    df_a = sample[mask].copy()
    df_a["alarm_type"] = "Under Voltage"
    df_a["severity"]   = np.where(df_a["voltage"] < 360, "Critical", "Warning")
    df_a["value"]      = df_a["voltage"].astype(str) + " V"
    alarms.append(df_a[["timestamp","machine_id","alarm_type","severity","value"]])

    # Over Voltage
    mask = sample["voltage"] > 392
    df_a = sample[mask].copy()
    df_a["alarm_type"] = "Over Voltage"
    df_a["severity"]   = np.where(df_a["voltage"] > 396, "Critical", "Warning")
    df_a["value"]      = df_a["voltage"].astype(str) + " V"
    alarms.append(df_a[["timestamp","machine_id","alarm_type","severity","value"]])

    # Low Power Factor (ATURAN 5 related)
    mask = sample["power_factor"] < 0.84
    df_a = sample[mask].copy()
    df_a["alarm_type"] = "Low Power Factor"
    df_a["severity"]   = np.where(df_a["power_factor"] < 0.80, "Critical", "Warning")
    df_a["value"]      = df_a["power_factor"].astype(str)
    alarms.append(df_a[["timestamp","machine_id","alarm_type","severity","value"]])

    # Over Pressure
    mask = (sample["pressure"] > 11.0) & (sample["pressure"] > 0)
    df_a = sample[mask].copy()
    df_a["alarm_type"] = "Over Pressure"
    df_a["severity"]   = np.where(df_a["pressure"] > 12.0, "Critical", "Warning")
    df_a["value"]      = df_a["pressure"].astype(str) + " Bar"
    alarms.append(df_a[["timestamp","machine_id","alarm_type","severity","value"]])

    if alarms:
        alarm_df = pd.concat(alarms, ignore_index=True).sort_values("timestamp")
        alarm_df.insert(0, "alarm_id", [f"ALM-{i:05d}" for i in range(1, len(alarm_df)+1)])
        alarm_df.to_csv(os.path.join(RAW, "alarm_history.csv"), index=False)
        print(f"      ✓ {len(alarm_df):,} alarm disimpan.")
        return alarm_df
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────
# STEP 4 — MAINTENANCE HISTORY
# ─────────────────────────────────────────────────────────
MAINTENANCE_TYPES = [
    "Preventive Maintenance",
    "Corrective Maintenance – Bearing Replacement",
    "Corrective Maintenance – Seal Replacement",
    "Corrective Maintenance – Motor Winding",
    "Calibration",
    "Lubrication Service",
    "Belt Replacement",
    "Impeller Inspection",
    "Cooling Fan Cleaning",
    "Electrical Inspection",
]
TECHNICIANS = ["Budi S.", "Hendra K.", "Wahyu P.", "Rian F.", "Dewi A.", "Andri M.", "Siti R."]

def gen_maintenance(machines_df, sensor_log):
    print("\n[4/6] Maintenance History...")
    records = []
    maint_id = 1
    rng = np.random.default_rng(SEED + 1)

    for _, row in machines_df.iterrows():
        m = row.to_dict()
        age = CURRENT_YEAR - m["install_year"]
        # Preventive: setiap 3 bulan = 4 kali per tahun
        # Corrective: tergantung usia
        n_preventive = 4
        n_corrective = max(1, int(age * 0.25) + rng.integers(0, 3))

        # Preventive (Q1, Q2, Q3, Q4)
        pm_months = [2, 5, 8, 11]
        for pm_month in pm_months:
            day = rng.integers(1, 25)
            date = datetime(2024, pm_month, int(day))
            records.append({
                "maintenance_id": f"MNT-{maint_id:04d}",
                "machine_id": m["machine_id"],
                "date": date.strftime("%Y-%m-%d"),
                "type": "Preventive Maintenance",
                "technician": rng.choice(TECHNICIANS),
                "duration_hours": round(float(rng.uniform(2, 6)), 1),
                "component": rng.choice(["Lubrication","Inspection","Filter Cleaning","Calibration"]),
                "cost_idr": int(rng.integers(500_000, 2_000_000)),
            })
            maint_id += 1

        # Corrective
        for _ in range(n_corrective):
            month = rng.integers(1, 13)
            day   = rng.integers(1, 27)
            date  = datetime(2024, int(month), int(day))
            mtype = rng.choice(MAINTENANCE_TYPES[1:])  # Skip preventive
            records.append({
                "maintenance_id": f"MNT-{maint_id:04d}",
                "machine_id": m["machine_id"],
                "date": date.strftime("%Y-%m-%d"),
                "type": mtype,
                "technician": rng.choice(TECHNICIANS),
                "duration_hours": round(float(rng.uniform(4, 24)), 1),
                "component": mtype.replace("Corrective Maintenance – ", ""),
                "cost_idr": int(rng.integers(1_500_000, 15_000_000)),
            })
            maint_id += 1

    maint_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    maint_df.to_csv(os.path.join(RAW, "maintenance_history.csv"), index=False)
    print(f"      ✓ {len(maint_df)} record maintenance disimpan.")
    return maint_df


# ─────────────────────────────────────────────────────────
# STEP 5 — MACHINE STATUS & HEALTH SCORE
# ─────────────────────────────────────────────────────────
def gen_machine_status(machines_df, sensor_log, alarm_df):
    print("\n[5/6] Machine Status & Health Score...")
    rng = np.random.default_rng(SEED + 2)

    # Ambil data 30 hari terakhir
    recent_cutoff = TIMESTAMPS[-1] - pd.Timedelta(days=30)
    recent = sensor_log[sensor_log["timestamp"] >= recent_cutoff]

    # Hitung rata-rata per mesin
    agg = recent.groupby("machine_id").agg(
        avg_temp   = ("temperature", "mean"),
        max_temp   = ("temperature", "max"),
        avg_vib    = ("vibration", "mean"),
        max_vib    = ("vibration", "max"),
        avg_pf     = ("power_factor", "mean"),
        avg_curr   = ("current", "mean"),
        avg_power  = ("power_kw", "mean"),
    ).reset_index()

    # Alarm count per mesin (30 hari)
    if len(alarm_df):
        alarm_count = alarm_df[alarm_df["timestamp"] >= recent_cutoff].groupby("machine_id").size().reset_index(name="alarm_count_30d")
        agg = agg.merge(alarm_count, on="machine_id", how="left")
    else:
        agg["alarm_count_30d"] = 0
    agg["alarm_count_30d"] = agg["alarm_count_30d"].fillna(0).astype(int)

    # Gabung dengan install_year
    agg = agg.merge(machines_df[["machine_id","install_year","machine_type"]], on="machine_id")

    # ── Health Score Algorithm ──────────────────────────────────────────
    def health_score(row):
        score = 100.0
        age = CURRENT_YEAR - row["install_year"]
        # Penalti usia (max -20)
        score -= min(20, age * 1.3)
        # Penalti suhu
        if row["max_temp"] > 80:   score -= 25
        elif row["max_temp"] > 70: score -= 12
        elif row["avg_temp"] > 65: score -= 5
        # Penalti vibrasi
        if row["max_vib"] > 4.5:   score -= 30
        elif row["max_vib"] > 3.5: score -= 15
        elif row["avg_vib"] > 2.8: score -= 5
        # Penalti power factor
        if row["avg_pf"] < 0.80:   score -= 15
        elif row["avg_pf"] < 0.83: score -= 7
        # Penalti alarm frekuensi
        score -= min(20, row["alarm_count_30d"] * 0.15)
        return max(0, min(100, round(score, 1)))

    agg["health_score"] = agg.apply(health_score, axis=1)

    def status_from_score(s):
        if s >= 75: return "Good"
        if s >= 50: return "Warning"
        return "Critical"

    agg["status"] = agg["health_score"].apply(status_from_score)

    # Tambahkan 2 mesin offline secara random
    offline_ids = rng.choice(agg["machine_id"].values, size=2, replace=False)
    agg.loc[agg["machine_id"].isin(offline_ids), "status"] = "Offline"

    status_df = agg[["machine_id","machine_type","status","health_score",
                      "avg_temp","avg_vib","avg_pf","alarm_count_30d"]].copy()
    status_df.columns = ["machine_id","machine_type","status","health_score",
                          "avg_temp_30d","avg_vib_30d","avg_pf_30d","alarm_count_30d"]
    status_df.to_csv(os.path.join(PROC, "machine_status.csv"), index=False)
    print(f"      ✓ Health score dihitung untuk {len(status_df)} mesin.")
    return status_df


# ─────────────────────────────────────────────────────────
# STEP 6 — PROCESSED AGGREGATES (untuk performa dashboard)
# ─────────────────────────────────────────────────────────
def gen_processed(sensor_log, machines_df):
    print("\n[6/6] Aggregates untuk Dashboard...")

    # Hourly aggregate
    sl = sensor_log.copy()
    sl["hour"] = sl["timestamp"].dt.floor("h")
    hourly = sl.groupby(["hour","machine_id"]).agg(
        voltage     = ("voltage","mean"),
        current     = ("current","mean"),
        temperature = ("temperature","mean"),
        vibration   = ("vibration","mean"),
        humidity    = ("humidity","mean"),
        pressure    = ("pressure","mean"),
        flow_rate   = ("flow_rate","mean"),
        frequency   = ("frequency","mean"),
        power_factor= ("power_factor","mean"),
        power_kw    = ("power_kw","mean"),
        energy_kwh  = ("energy_kwh","sum"),
    ).reset_index().rename(columns={"hour":"timestamp"})
    hourly.to_csv(os.path.join(PROC, "sensor_hourly.csv"), index=False)
    print(f"      ✓ Hourly: {len(hourly):,} record")

    # Daily aggregate
    sl["date"] = sl["timestamp"].dt.date
    daily = sl.groupby(["date","machine_id"]).agg(
        avg_voltage     = ("voltage","mean"),
        avg_current     = ("current","mean"),
        avg_temperature = ("temperature","mean"),
        max_temperature = ("temperature","max"),
        avg_vibration   = ("vibration","mean"),
        max_vibration   = ("vibration","max"),
        avg_humidity    = ("humidity","mean"),
        avg_pressure    = ("pressure","mean"),
        avg_flow        = ("flow_rate","mean"),
        avg_pf          = ("power_factor","mean"),
        avg_power_kw    = ("power_kw","mean"),
        total_energy_kwh= ("energy_kwh","sum"),
    ).reset_index()
    daily.to_csv(os.path.join(PROC, "sensor_daily.csv"), index=False)
    print(f"      ✓ Daily:  {len(daily):,} record")

    # Energy summary per machine per bulan
    sl["month"] = sl["timestamp"].dt.to_period("M").astype(str)
    energy = sl.groupby(["month","machine_id"])["energy_kwh"].sum().reset_index()
    energy = energy.merge(machines_df[["machine_id","machine_type","rated_power_kw"]], on="machine_id")
    energy.to_csv(os.path.join(PROC, "energy_monthly.csv"), index=False)
    print(f"      ✓ Energy monthly: {len(energy):,} record")

    # Operator shift (bonus)
    shifts = []
    ops = ["Budi S.", "Hendra K.", "Wahyu P.", "Rian F.", "Dewi A."]
    areas = ["Production Hall A", "Production Hall B", "Production Hall C",
             "Pump Station", "Compressor Room"]
    for op in ops:
        for area in areas:
            shifts.append({"operator": op,
                           "shift": np.random.choice(["Shift 1 (07-15)","Shift 2 (15-23)","Shift 3 (23-07)"]),
                           "area": area})
    pd.DataFrame(shifts).to_csv(os.path.join(MASTER, "operator_shift.csv"), index=False)

    # Machine master copy to processed
    machines_df[["machine_id","machine_type","location","rated_power_kw","install_year"]].to_csv(
        os.path.join(PROC, "machine_master.csv"), index=False)
    print("      ✓ Master & shift disimpan.")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Smart Industrial IoT — Data Generator")
    print("  PT Nusantara Steel Manufacturing (Simulation)")
    print("=" * 60)
    print(f"  Periode  : {START_DATE.date()} → {END_DATE.date()}")
    print(f"  Interval : {INTERVAL}")
    print(f"  Mesin    : {len(MACHINES)} unit")
    print(f"  Target   : {N_STEPS * len(MACHINES):,} records")
    print("=" * 60)

    machines_df = gen_machine_master()
    sensor_log, fault_events = gen_sensor_log(machines_df)
    alarm_df   = gen_alarm_history(sensor_log, machines_df)
    maint_df   = gen_maintenance(machines_df, sensor_log)
    status_df  = gen_machine_status(machines_df, sensor_log, alarm_df)
    gen_processed(sensor_log, machines_df)

    print("\n" + "=" * 60)
    print("  SELESAI! Ringkasan:")
    print(f"  • sensor_log.csv      : {len(sensor_log):>10,} records")
    print(f"  • alarm_history.csv   : {len(alarm_df):>10,} records")
    print(f"  • maintenance_history : {len(maint_df):>10,} records")
    print(f"  • machine_status.csv  : {len(status_df):>10,} records")
    print("=" * 60)
    print("\n  Langkah selanjutnya:")
    print("  1. cd dashboard")
    print("  2. streamlit run Home.py")
    print("=" * 60)

if __name__ == "__main__":
    main()

"""
Phase 2 & 3: Data Cleaning + EDA
PT Surya Textile Indonesia - Energy Optimization Project
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
RAW_PATH     = "/home/claude/Energy_Cost_Optimization/data/raw/energy_consumption.csv"
CLEAN_PATH   = "/home/claude/Energy_Cost_Optimization/data/processed/clean_energy_data.csv"
REPORTS_PATH = "/home/claude/Energy_Cost_Optimization/reports/"

# ─── Colour palette (corporate style) ────────────────────────────────────────
BLUE    = "#1B3A6B"
ORANGE  = "#E87722"
GREEN   = "#2E7D32"
RED     = "#C62828"
GRAY    = "#607D8B"
LBLUE   = "#90CAF9"
palette = [BLUE, ORANGE, GREEN, RED, GRAY, LBLUE,
           "#7B1FA2", "#F57F17", "#00695C", "#AD1457"]

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F9FA",
    "axes.edgecolor":    "#DEE2E6",
    "grid.color":        "#DEE2E6",
    "grid.linestyle":    "--",
    "grid.alpha":        0.7,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titleweight":  "bold",
    "axes.titlesize":    12,
})

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 2: DATA CLEANING")
print("=" * 60)

df = pd.read_csv(RAW_PATH, parse_dates=["timestamp"])
print(f"Raw shape       : {df.shape}")

cleaning_log = {}

# 1. Missing values
mv = df.isnull().sum()
cleaning_log["missing_before"] = int(mv.sum())
df.dropna(inplace=True)
print(f"Missing values  : {cleaning_log['missing_before']} → 0 after drop")

# 2. Duplicates
dup = df.duplicated(subset=["timestamp", "machine_id"]).sum()
cleaning_log["duplicates"] = int(dup)
df.drop_duplicates(subset=["timestamp", "machine_id"], inplace=True)
print(f"Duplicates      : {dup} removed")

# 3. Outliers – IQR per column
numeric_cols = ["voltage_v", "current_a", "power_factor",
                "active_power_kw", "reactive_power_kvar",
                "apparent_power_kva", "energy_kwh"]
outlier_counts = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 3.0 * IQR, Q3 + 3.0 * IQR
    mask = (df[col] < lo) | (df[col] > hi)
    outlier_counts[col] = int(mask.sum())
    df = df[~mask]
cleaning_log["outliers"] = outlier_counts
total_outliers = sum(outlier_counts.values())
print(f"Outliers removed: {total_outliers:,} (3×IQR rule)")

# 4. Power-factor range guard
invalid_pf = ((df["power_factor"] < 0.5) | (df["power_factor"] > 1.0)).sum()
df = df[(df["power_factor"] >= 0.5) & (df["power_factor"] <= 1.0)]
print(f"Invalid PF rows : {invalid_pf} removed")

# 5. Computed cost column
df["cost_idr"] = df["energy_kwh"] * df["tariff_idr_kwh"]

# 6. Helper columns
df["hour"]    = df["timestamp"].dt.hour
df["date"]    = df["timestamp"].dt.date
df["day_of_week"] = df["timestamp"].dt.day_name()
df["month"]   = df["timestamp"].dt.month
df["is_peak"] = df["hour"].apply(lambda h: 1 if 17 <= h < 22 else 0)

df.to_csv(CLEAN_PATH, index=False)
cleaning_log["clean_shape"] = df.shape
print(f"\nClean shape     : {df.shape}")
print(f"✅ Clean data saved → {CLEAN_PATH}\n")

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ── Aggregations ──────────────────────────────────────────────────────────────
machine_summary = (df.groupby(["machine_id", "machine_type"])
                     .agg(total_energy_kwh=("energy_kwh", "sum"),
                          total_cost_idr=("cost_idr", "sum"),
                          avg_power_factor=("power_factor", "mean"),
                          avg_power_kw=("active_power_kw", "mean"))
                     .reset_index()
                     .sort_values("total_energy_kwh", ascending=False))

hourly_summary = (df.groupby("hour")
                    .agg(avg_energy=("energy_kwh", "mean"),
                         avg_cost=("cost_idr", "mean"),
                         total_cost=("cost_idr", "sum"))
                    .reset_index())

daily_summary = (df.groupby("day_of_week")
                   .agg(avg_cost=("cost_idr", "mean"),
                        total_cost=("cost_idr", "sum"))
                   .reset_index())

day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
daily_summary["day_of_week"] = pd.Categorical(daily_summary["day_of_week"],
                                               categories=day_order, ordered=True)
daily_summary.sort_values("day_of_week", inplace=True)

pf_summary = (machine_summary.sort_values("avg_power_factor").head(10))

top10_energy = machine_summary.head(10)
top10_cost   = machine_summary.sort_values("total_cost_idr", ascending=False).head(10)

print(f"Top consumer    : {top10_energy.iloc[0]['machine_id']} "
      f"({top10_energy.iloc[0]['total_energy_kwh']:,.0f} kWh)")
print(f"Highest cost    : {top10_cost.iloc[0]['machine_id']} "
      f"(Rp {top10_cost.iloc[0]['total_cost_idr']:,.0f})")
print(f"Worst PF machine: {pf_summary.iloc[0]['machine_id']} "
      f"(PF = {pf_summary.iloc[0]['avg_power_factor']:.4f})")

# ── Save figure images ─────────────────────────────────────────────────────────
FIG_DIR = REPORTS_PATH + "figures/"
import os; os.makedirs(FIG_DIR, exist_ok=True)

def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR + name, dpi=150, bbox_inches="tight")
    plt.close()

# Q1 – Top 10 Energy Consumers ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = [palette[{"Motor":0,"Pump":1,"Compressor":2}[t]]
          for t in top10_energy["machine_type"]]
bars = ax.barh(top10_energy["machine_id"][::-1],
               top10_energy["total_energy_kwh"][::-1],
               color=colors[::-1], edgecolor="white", linewidth=0.5)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 100, bar.get_y() + bar.get_height() / 2,
            f"{w:,.0f} kWh", va="center", fontsize=9)
legend_handles = [
    mpatches.Patch(color=BLUE,   label="Motor"),
    mpatches.Patch(color=ORANGE, label="Pump"),
    mpatches.Patch(color=GREEN,  label="Compressor"),
]
ax.legend(handles=legend_handles, loc="lower right")
ax.set_title("Q1 – Top 10 Machines by Energy Consumption (2025)")
ax.set_xlabel("Total Energy Consumption (kWh)")
ax.grid(axis="x")
save_fig("q1_top10_energy.png")

# Q2 – Top 10 Highest Cost ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors2 = [palette[{"Motor":0,"Pump":1,"Compressor":2}[t]]
           for t in top10_cost["machine_type"]]
bars2 = ax.barh(top10_cost["machine_id"][::-1],
                top10_cost["total_cost_idr"][::-1] / 1e6,
                color=colors2[::-1], edgecolor="white", linewidth=0.5)
for bar in bars2:
    w = bar.get_width()
    ax.text(w + 0.2, bar.get_y() + bar.get_height() / 2,
            f"Rp {w:.1f}M", va="center", fontsize=9)
ax.set_title("Q2 – Top 10 Machines by Electricity Cost (2025)")
ax.set_xlabel("Total Cost (IDR Juta)")
ax.grid(axis="x")
save_fig("q2_top10_cost.png")

# Q3 – Hourly Cost Profile ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
bar_colors = [RED if 17 <= h < 22 else BLUE for h in hourly_summary["hour"]]
ax.bar(hourly_summary["hour"], hourly_summary["avg_cost"],
       color=bar_colors, edgecolor="white", linewidth=0.5, width=0.85)
ax.axvspan(16.5, 21.5, alpha=0.08, color=RED, label="Peak Hours (17:00–22:00)")
ax.set_title("Q3 – Average Cost per Hour (IDR/interval)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Avg Cost (IDR)")
ax.set_xticks(range(0, 24))
peak_patch  = mpatches.Patch(color=RED,  alpha=0.6, label="Peak Hour")
normal_patch = mpatches.Patch(color=BLUE, label="Normal Hour")
ax.legend(handles=[peak_patch, normal_patch])
ax.grid(axis="y")
save_fig("q3_hourly_cost.png")

# Q4 – Daily Cost ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(daily_summary["day_of_week"], daily_summary["avg_cost"],
       color=palette[:7], edgecolor="white", linewidth=0.5)
ax.set_title("Q4 – Average Cost by Day of Week")
ax.set_xlabel("Day of Week")
ax.set_ylabel("Avg Cost (IDR/interval)")
ax.grid(axis="y")
plt.xticks(rotation=30)
save_fig("q4_daily_cost.png")

# Q5 – Power Factor Ranking ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
pf_colors = ["#C62828" if pf < 0.80 else "#2E7D32"
             for pf in pf_summary["avg_power_factor"]]
ax.barh(pf_summary["machine_id"][::-1],
        pf_summary["avg_power_factor"][::-1],
        color=pf_colors[::-1], edgecolor="white", linewidth=0.5)
ax.axvline(0.85, color="orange", linestyle="--", linewidth=1.5, label="Target PF = 0.85")
for i, (_, row) in enumerate(pf_summary.iloc[::-1].iterrows()):
    ax.text(row["avg_power_factor"] + 0.003,
            i, f"{row['avg_power_factor']:.4f}", va="center", fontsize=9)
ax.set_title("Q5 – Worst 10 Machines by Power Factor")
ax.set_xlabel("Average Power Factor")
ax.set_xlim(0.55, 1.0)
ax.legend()
ax.grid(axis="x")
save_fig("q5_power_factor.png")

# PF Distribution ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["power_factor"], bins=50, color=BLUE, edgecolor="white",
        linewidth=0.3, alpha=0.85)
ax.axvline(0.85, color=RED, linestyle="--", linewidth=1.5, label="PLN Minimum PF = 0.85")
ax.set_title("Power Factor Distribution – All Machines")
ax.set_xlabel("Power Factor")
ax.set_ylabel("Frequency")
ax.legend()
ax.grid(axis="y")
save_fig("pf_distribution.png")

# Monthly Energy Trend ──────────────────────────────────────────────────────────
monthly = (df.groupby("month")
             .agg(total_energy=("energy_kwh", "sum"),
                  total_cost=("cost_idr", "sum"))
             .reset_index())
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
monthly["month_name"] = [month_names[m-1] for m in monthly["month"]]

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
ax1.bar(monthly["month_name"], monthly["total_energy"]/1000,
        color=BLUE, alpha=0.75, label="Energy (MWh)")
ax2.plot(monthly["month_name"], monthly["total_cost"]/1e9,
         color=ORANGE, linewidth=2.5, marker="o", label="Cost (Rp Milyar)")
ax1.set_ylabel("Energy (MWh)", color=BLUE)
ax2.set_ylabel("Cost (Rp Milyar)", color=ORANGE)
ax1.set_title("Monthly Energy & Cost Trend (2025)")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
ax1.grid(axis="y", alpha=0.5)
save_fig("monthly_trend.png")

# Machine type comparison ────────────────────────────────────────────────────────
type_summary = (machine_summary.groupby("machine_type")
                               .agg(total_energy=("total_energy_kwh","sum"),
                                    total_cost=("total_cost_idr","sum"),
                                    avg_pf=("avg_power_factor","mean"))
                               .reset_index())

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, label, color in zip(
        axes,
        ["total_energy", "total_cost", "avg_pf"],
        ["Total Energy (kWh)", "Total Cost (IDR)", "Avg Power Factor"],
        [BLUE, ORANGE, GREEN]):
    vals = type_summary[col]
    if col == "total_cost":
        bars = ax.bar(type_summary["machine_type"], vals/1e9, color=color,
                      edgecolor="white")
        ax.set_ylabel("IDR Milyar")
    elif col == "total_energy":
        bars = ax.bar(type_summary["machine_type"], vals/1000, color=color,
                      edgecolor="white")
        ax.set_ylabel("MWh")
    else:
        bars = ax.bar(type_summary["machine_type"], vals, color=color,
                      edgecolor="white")
        ax.axhline(0.85, color=RED, linestyle="--", linewidth=1.2)
        ax.set_ylabel("Power Factor")
    ax.set_title(label)
    ax.grid(axis="y")
save_fig("machine_type_comparison.png")

print("\n✅ All EDA charts saved to reports/figures/")
print(f"\nKey findings:")
print(f"  Total energy 2025 : {df['energy_kwh'].sum():,.0f} kWh")
print(f"  Total cost   2025 : Rp {df['cost_idr'].sum():,.0f}")
print(f"  Peak hour avg cost: Rp {df[df['is_peak']==1]['cost_idr'].mean():,.2f}/interval")
print(f"  Normal hour avg   : Rp {df[df['is_peak']==0]['cost_idr'].mean():,.2f}/interval")

# Save KPIs for report
import json
kpis = {
    "total_energy_kwh": float(df["energy_kwh"].sum()),
    "total_cost_idr":   float(df["cost_idr"].sum()),
    "avg_daily_cost":   float(df.groupby("date")["cost_idr"].sum().mean()),
    "peak_pct":         float((df["is_peak"] == 1).mean() * 100),
    "bad_pf_pct":       float((df["power_factor"] < 0.80).mean() * 100),
    "machine_summary":  machine_summary.to_dict(orient="records"),
    "hourly_summary":   hourly_summary.to_dict(orient="records"),
    "daily_summary":    daily_summary.to_dict(orient="records"),
    "type_summary":     type_summary.to_dict(orient="records"),
    "top10_energy":     top10_energy.to_dict(orient="records"),
    "top10_cost":       top10_cost.to_dict(orient="records"),
    "pf_worst10":       pf_summary.to_dict(orient="records"),
}
with open(REPORTS_PATH + "kpis.json", "w") as f:
    json.dump(kpis, f, indent=2, default=str)
print(f"\n✅ KPIs saved → reports/kpis.json")

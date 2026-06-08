"""
Phase 4 & 5: Cost Analysis + Energy Saving Simulation
PT Surya Textile Indonesia - Energy Optimization Project
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import warnings
warnings.filterwarnings("ignore")

CLEAN_PATH   = "/home/claude/Energy_Cost_Optimization/data/processed/clean_energy_data.csv"
REPORTS_PATH = "/home/claude/Energy_Cost_Optimization/reports/"
FIG_DIR      = REPORTS_PATH + "figures/"

BLUE   = "#1B3A6B"
ORANGE = "#E87722"
GREEN  = "#2E7D32"
RED    = "#C62828"
GRAY   = "#607D8B"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F9FA",
    "axes.edgecolor":   "#DEE2E6",
    "grid.color":       "#DEE2E6",
    "grid.linestyle":   "--",
    "grid.alpha":       0.7,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titleweight": "bold",
    "axes.titlesize":   12,
})

print("=" * 60)
print("PHASE 4: COST ANALYSIS")
print("=" * 60)

df = pd.read_csv(CLEAN_PATH, parse_dates=["timestamp"])

# ─── Phase 4: KPIs ────────────────────────────────────────────────────────────
total_energy_kwh  = df["energy_kwh"].sum()
total_cost_idr    = df["cost_idr"].sum()
avg_daily_cost    = df.groupby("date")["cost_idr"].sum().mean()
avg_monthly_cost  = df.groupby("month")["cost_idr"].sum().mean()
peak_cost         = df[df["is_peak"] == 1]["cost_idr"].sum()
normal_cost       = df[df["is_peak"] == 0]["cost_idr"].sum()

machine_cost = (df.groupby(["machine_id", "machine_type"])
                  .agg(total_energy=("energy_kwh","sum"),
                       total_cost=("cost_idr","sum"),
                       avg_pf=("power_factor","mean"))
                  .reset_index()
                  .sort_values("total_cost", ascending=False))

print(f"Total Energy     : {total_energy_kwh:,.0f} kWh")
print(f"Total Cost 2025  : Rp {total_cost_idr:,.0f}")
print(f"Avg Daily Cost   : Rp {avg_daily_cost:,.0f}")
print(f"Avg Monthly Cost : Rp {avg_monthly_cost:,.0f}")
print(f"Peak Cost Share  : {peak_cost/total_cost_idr*100:.1f}%")
print(f"Normal Cost Share: {normal_cost/total_cost_idr*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 5: ENERGY SAVING SIMULATION")
print("=" * 60)

# ──── SCENARIO A: Power Factor Correction to 0.95 ─────────────────────────────
# Bad PF records (PF < 0.80)
bad_pf_mask = df["power_factor"] < 0.80

# If PF improves from current to 0.95, apparent power stays same but
# active power increases → energy per kVA improves
# Cost saving comes from: reactive power penalty reduction + lower apparent power billing
# We simulate: new_active_power = apparent_power * 0.95
df_bad = df[bad_pf_mask].copy()
original_active   = df_bad["active_power_kw"].sum()
corrected_active  = df_bad["apparent_power_kva"] * 0.95
delta_active      = corrected_active.sum() - original_active

# Energy saving: machines now produce same work with better PF
# Reactive energy wasted is reduced → cost saving modeled as:
#   For each bad-PF interval: saving = (S × (0.95 - PF_current)) × 0.25 × tariff
df_bad_copy = df_bad.copy()
df_bad_copy["energy_saving_kwh"] = (
    df_bad_copy["apparent_power_kva"] * (0.95 - df_bad_copy["power_factor"]) * 0.25
)
df_bad_copy["cost_saving_idr"] = (
    df_bad_copy["energy_saving_kwh"] * df_bad_copy["tariff_idr_kwh"]
)
saving_a_kwh = df_bad_copy["energy_saving_kwh"].sum()
saving_a_idr = df_bad_copy["cost_saving_idr"].sum()
saving_a_pct = saving_a_idr / total_cost_idr * 100

# Per machine breakdown for Scenario A
saving_a_by_machine = (df_bad_copy.groupby("machine_id")
                                  .agg(saving_kwh=("energy_saving_kwh","sum"),
                                       saving_idr=("cost_saving_idr","sum"))
                                  .sort_values("saving_idr", ascending=False)
                                  .head(10))

print(f"\n📌 Scenario A – PF Correction to 0.95:")
print(f"   Bad PF records       : {bad_pf_mask.sum():,} ({bad_pf_mask.mean()*100:.1f}%)")
print(f"   Energy saving        : {saving_a_kwh:,.0f} kWh")
print(f"   Cost saving (annual) : Rp {saving_a_idr:,.0f}")
print(f"   Saving percentage    : {saving_a_pct:.2f}%")

# ──── SCENARIO B: Shift peak load 18:00→13:00 ─────────────────────────────────
# Records currently at hour=18 (peak, tariff 1700)
peak_18_mask = df["hour"] == 18
df_peak18    = df[peak_18_mask].copy()

# If shifted to 13:00 (normal hour, tariff 1450)
original_cost_18  = df_peak18["cost_idr"].sum()
new_cost_13       = df_peak18["energy_kwh"].sum() * 1450
saving_b_idr      = original_cost_18 - new_cost_13
saving_b_kwh      = df_peak18["energy_kwh"].sum()
saving_b_pct      = saving_b_idr / total_cost_idr * 100

print(f"\n📌 Scenario B – Shift Load 18:00 → 13:00:")
print(f"   Affected intervals   : {peak_18_mask.sum():,}")
print(f"   Energy affected      : {saving_b_kwh:,.0f} kWh")
print(f"   Original cost (1700) : Rp {original_cost_18:,.0f}")
print(f"   New cost (1450)      : Rp {new_cost_13:,.0f}")
print(f"   Cost saving (annual) : Rp {saving_b_idr:,.0f}")
print(f"   Saving percentage    : {saving_b_pct:.2f}%")

# ──── SCENARIO C: 5% energy reduction ─────────────────────────────────────────
reduction_5pct_kwh = total_energy_kwh * 0.05
saving_c_idr       = total_cost_idr * 0.05
saving_c_pct       = 5.0

print(f"\n📌 Scenario C – 5% Energy Reduction:")
print(f"   Energy reduction     : {reduction_5pct_kwh:,.0f} kWh")
print(f"   Cost saving (annual) : Rp {saving_c_idr:,.0f}")
print(f"   Saving percentage    : {saving_c_pct:.2f}%")

# ──── Combined Scenario ────────────────────────────────────────────────────────
# (avoiding double-counting: take maximum practical saving)
combined_saving_idr = saving_a_idr + saving_b_idr + saving_c_idr
combined_saving_pct = combined_saving_idr / total_cost_idr * 100
print(f"\n📌 Combined (A + B + C) Potential Saving:")
print(f"   Total saving (annual): Rp {combined_saving_idr:,.0f}")
print(f"   Saving percentage    : {combined_saving_pct:.2f}%")

# ─── Visualisations ────────────────────────────────────────────────────────────
def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR + name, dpi=150, bbox_inches="tight")
    plt.close()

# Cost Breakdown by Category
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Pie: Peak vs Normal
axes[0].pie(
    [peak_cost, normal_cost],
    labels=["Peak (17–22 hrs)", "Normal Hours"],
    colors=[RED, BLUE], autopct="%1.1f%%",
    startangle=140, pctdistance=0.75,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
)
axes[0].set_title("Cost Split: Peak vs Normal Hours")

# Bar: by machine type
type_cost = (df.groupby("machine_type")["cost_idr"].sum().reset_index())
bars = axes[1].bar(type_cost["machine_type"], type_cost["cost_idr"] / 1e9,
                   color=[BLUE, ORANGE, GREEN], edgecolor="white")
for bar in bars:
    w = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, w + 0.01,
                 f"Rp {w:.2f}B", ha="center", va="bottom", fontsize=10)
axes[1].set_title("Total Cost by Machine Type (IDR Milyar)")
axes[1].set_ylabel("IDR Milyar")
axes[1].grid(axis="y")
save_fig("cost_breakdown.png")

# Scenario comparison chart
fig, ax = plt.subplots(figsize=(10, 5))
scenarios = ["Current Cost", "Scenario A\n(PF Correction)",
             "Scenario B\n(Load Shift 18→13)", "Scenario C\n(5% Reduction)",
             "Combined\n(A+B+C)"]
values = [
    total_cost_idr / 1e9,
    (total_cost_idr - saving_a_idr) / 1e9,
    (total_cost_idr - saving_b_idr) / 1e9,
    (total_cost_idr - saving_c_idr) / 1e9,
    (total_cost_idr - combined_saving_idr) / 1e9,
]
colors = [GRAY, GREEN, ORANGE, BLUE, RED]
bars = ax.bar(scenarios, values, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
ax.axhline(total_cost_idr / 1e9, color=GRAY, linestyle="--",
           linewidth=1.2, label="Current Baseline")
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
            f"Rp {v:.2f}B", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_title("Energy Cost Comparison: Saving Scenarios vs Current (2025)")
ax.set_ylabel("Annual Cost (IDR Milyar)")
ax.grid(axis="y")
save_fig("scenario_comparison.png")

# Top 10 machines for Scenario A savings
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(saving_a_by_machine.index[::-1],
        saving_a_by_machine["saving_idr"][::-1] / 1e6,
        color=GREEN, edgecolor="white")
for i, (mid, row) in enumerate(saving_a_by_machine.iloc[::-1].iterrows()):
    ax.text(row["saving_idr"]/1e6 + 0.1, i,
            f"Rp {row['saving_idr']/1e6:.2f}M", va="center", fontsize=9)
ax.set_title("Scenario A – Potential Savings per Machine (PF Correction to 0.95)")
ax.set_xlabel("Annual Saving (IDR Juta)")
ax.grid(axis="x")
save_fig("scenario_a_by_machine.png")

# Monthly cost with and without savings
monthly_cost = df.groupby("month")["cost_idr"].sum()
month_names  = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
fig, ax = plt.subplots(figsize=(12, 5))
x = range(len(monthly_cost))
ax.bar([i - 0.2 for i in x], monthly_cost.values / 1e6, width=0.38,
       color=BLUE, label="Current Cost", edgecolor="white")
ax.bar([i + 0.2 for i in x],
       (monthly_cost.values * (1 - combined_saving_pct/100)) / 1e6,
       width=0.38, color=GREEN, label="After Combined Savings", edgecolor="white")
ax.set_xticks(list(x))
ax.set_xticklabels(month_names)
ax.set_title("Monthly Cost: Current vs. After Combined Savings")
ax.set_ylabel("Cost (IDR Juta)")
ax.legend()
ax.grid(axis="y")
save_fig("monthly_cost_comparison.png")

print("\n✅ All scenario charts saved")

# ─── Save simulation results ───────────────────────────────────────────────────
simulation = {
    "baseline": {
        "total_energy_kwh": total_energy_kwh,
        "total_cost_idr":   total_cost_idr,
        "avg_daily_cost":   avg_daily_cost,
        "peak_cost_idr":    peak_cost,
        "normal_cost_idr":  normal_cost,
    },
    "scenario_a": {
        "name":          "Power Factor Correction to 0.95",
        "saving_kwh":    saving_a_kwh,
        "saving_idr":    saving_a_idr,
        "saving_pct":    saving_a_pct,
        "new_annual_cost": total_cost_idr - saving_a_idr,
    },
    "scenario_b": {
        "name":          "Load Shift 18:00 → 13:00",
        "saving_kwh":    saving_b_kwh,
        "saving_idr":    saving_b_idr,
        "saving_pct":    saving_b_pct,
        "new_annual_cost": total_cost_idr - saving_b_idr,
    },
    "scenario_c": {
        "name":          "5% Energy Consumption Reduction",
        "saving_kwh":    reduction_5pct_kwh,
        "saving_idr":    saving_c_idr,
        "saving_pct":    saving_c_pct,
        "new_annual_cost": total_cost_idr - saving_c_idr,
    },
    "combined": {
        "name":          "Combined Scenarios A + B + C",
        "saving_idr":    combined_saving_idr,
        "saving_pct":    combined_saving_pct,
        "new_annual_cost": total_cost_idr - combined_saving_idr,
    },
    "machine_cost_top10": machine_cost.head(10).to_dict(orient="records"),
}
with open(REPORTS_PATH + "simulation.json", "w") as f:
    json.dump(simulation, f, indent=2, default=str)
print("✅ Simulation results saved → reports/simulation.json")

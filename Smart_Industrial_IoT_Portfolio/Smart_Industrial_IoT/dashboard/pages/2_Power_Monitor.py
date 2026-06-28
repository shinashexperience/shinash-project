"""
Page 3 — Power Monitor
Voltage, Current, Power, Power Factor, Frequency
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="Power Monitor | IoT Dashboard", page_icon="⚡", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0d1117; }
[data-testid="stSidebar"] { background-color: #161b22; }
body, p, span, div { color: #e6edf3; }
.section-title {
    font-size:15px; font-weight:600; color:#58a6ff;
    border-left:3px solid #f0b429; padding-left:10px; margin:18px 0 10px;
}
.kpi { background:#161b22; border:1px solid #30363d; border-radius:10px;
       padding:14px 16px; text-align:center; }
.kv  { font-size:28px; font-weight:700; }
.kl  { font-size:11px; color:#8b949e; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC   = os.path.join(BASE, "data", "processed")

@st.cache_data(ttl=300)
def load():
    daily  = pd.read_csv(os.path.join(PROC, "sensor_daily.csv"))
    daily["date"] = pd.to_datetime(daily["date"])
    hourly = pd.read_csv(os.path.join(PROC, "sensor_hourly.csv"))
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])
    master = pd.read_csv(os.path.join(PROC, "machine_master.csv"))
    energy = pd.read_csv(os.path.join(PROC, "energy_monthly.csv"))
    return daily, hourly, master, energy

daily, hourly, master, energy = load()

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Power Monitor")
    sel_machine = st.selectbox("Pilih Mesin", sorted(master["machine_id"].tolist()))
    date_max = daily["date"].max().date()
    date_min = daily["date"].min().date()
    sel_start = st.date_input("Mulai", date_max - pd.Timedelta(days=29),
                               min_value=date_min, max_value=date_max)
    sel_end   = st.date_input("Akhir", date_max,
                               min_value=date_min, max_value=date_max)
    granularity = st.radio("Granularitas", ["Harian","Jam-an"], index=0)
    st.divider()
    st.page_link("Home.py", label="← Kembali ke Home")

d_start = pd.Timestamp(sel_start)
d_end   = pd.Timestamp(sel_end)

m_info = master[master["machine_id"] == sel_machine].iloc[0]

# ── Header ───────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,#1c1a08,#111806);border-radius:12px;
     padding:20px 28px;margin-bottom:18px;border:1px solid #374151;">
  <h2 style="color:#e6edf3;margin:0;font-size:22px;">⚡ Power Monitor — {sel_machine}</h2>
  <p style="color:#9ca3af;margin:4px 0 0;font-size:13px;">
    Tipe: <b style="color:#f0b429">{m_info['machine_type']}</b> &nbsp;|&nbsp;
    Daya Nominal: {m_info['rated_power_kw']} kW &nbsp;|&nbsp;
    Lokasi: {m_info['location']}
  </p>
</div>
""", unsafe_allow_html=True)

# ── Filter ───────────────────────────────────────────────
if granularity == "Harian":
    df = daily[(daily["machine_id"] == sel_machine) &
               (daily["date"] >= d_start) & (daily["date"] <= d_end)].copy()
    time_col = "date"
    v_col, i_col, pf_col, pwr_col, freq_col = \
        "avg_voltage","avg_current","avg_pf","avg_power_kw","avg_voltage"  # proxy
else:
    df = hourly[(hourly["machine_id"] == sel_machine) &
                (hourly["timestamp"] >= d_start) & (hourly["timestamp"] <= d_end)].copy()
    time_col = "timestamp"
    v_col, i_col, pf_col, pwr_col = "voltage","current","power_factor","power_kw"

# ── KPI Row ──────────────────────────────────────────────
if granularity == "Harian":
    avg_v  = df["avg_voltage"].mean()   if len(df) else 0
    avg_i  = df["avg_current"].mean()   if len(df) else 0
    avg_pf = df["avg_pf"].mean()        if len(df) else 0
    avg_pw = df["avg_power_kw"].mean()  if len(df) else 0
    tot_e  = df["total_energy_kwh"].sum() if len(df) else 0
else:
    avg_v  = df["voltage"].mean()       if len(df) else 0
    avg_i  = df["current"].mean()       if len(df) else 0
    avg_pf = df["power_factor"].mean()  if len(df) else 0
    avg_pw = df["power_kw"].mean()      if len(df) else 0
    tot_e  = df["energy_kwh"].sum()     if len(df) else 0

k1,k2,k3,k4,k5 = st.columns(5)
def kpi(col, v, lbl, color="#e6edf3", sfx=""):
    col.markdown(f'<div class="kpi"><div class="kv" style="color:{color}">{v}{sfx}</div>'
                 f'<div class="kl">{lbl}</div></div>', unsafe_allow_html=True)

pf_color = "#2ea043" if avg_pf > 0.85 else ("#d29922" if avg_pf > 0.80 else "#f85149")
kpi(k1, f"{avg_v:.1f}",   "Avg Voltage (V)",        "#58a6ff")
kpi(k2, f"{avg_i:.1f}",   "Avg Current (A)",        "#f0b429")
kpi(k3, f"{avg_pf:.3f}",  "Avg Power Factor",       pf_color)
kpi(k4, f"{avg_pw:.1f}",  "Avg Power (kW)",         "#a371f7")
kpi(k5, f"{tot_e:,.0f}",  "Total Energi (kWh)",     "#2ea043")

st.markdown("")

# ── Voltage & Current Chart ───────────────────────────────
st.markdown('<div class="section-title">⚡ Tegangan & Arus</div>', unsafe_allow_html=True)

fig_vc = make_subplots(specs=[[{"secondary_y": True}]])
if granularity == "Harian":
    fig_vc.add_trace(go.Scatter(x=df["date"],y=df["avg_voltage"],
        name="Voltage (V)", line=dict(color="#58a6ff",width=2)), secondary_y=False)
    fig_vc.add_trace(go.Scatter(x=df["date"],y=df["avg_current"],
        name="Current (A)", line=dict(color="#f0b429",width=2)), secondary_y=True)
else:
    fig_vc.add_trace(go.Scatter(x=df["timestamp"],y=df["voltage"],
        name="Voltage (V)", line=dict(color="#58a6ff",width=1.5)), secondary_y=False)
    fig_vc.add_trace(go.Scatter(x=df["timestamp"],y=df["current"],
        name="Current (A)", line=dict(color="#f0b429",width=1.5)), secondary_y=True)

fig_vc.add_hline(y=395, line_dash="dash", line_color="#f85149",
                  annotation_text="Over Voltage 395V", annotation_font_size=10)
fig_vc.add_hline(y=365, line_dash="dash", line_color="#d29922",
                  annotation_text="Under Voltage 365V", annotation_font_size=10)
fig_vc.update_yaxes(title_text="Voltage (V)", secondary_y=False,
                     ticksuffix=" V", gridcolor="#21262d")
fig_vc.update_yaxes(title_text="Current (A)", secondary_y=True)
fig_vc.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    template="plotly_dark", height=280, margin=dict(l=0,r=0,t=10,b=0),
    legend=dict(orientation="h", y=1.05),
)
fig_vc.update_xaxes(gridcolor="#21262d")
st.plotly_chart(fig_vc, use_container_width=True)

# ── Power & Power Factor ──────────────────────────────────
cola, colb = st.columns(2)

with cola:
    st.markdown('<div class="section-title">🔋 Daya (kW)</div>', unsafe_allow_html=True)
    fig_pw = go.Figure()
    if granularity == "Harian":
        fig_pw.add_trace(go.Bar(
            x=df["date"], y=df["avg_power_kw"],
            marker_color="#a371f7", name="Avg Power kW",
        ))
    else:
        fig_pw.add_trace(go.Scatter(
            x=df["timestamp"], y=df["power_kw"],
            fill="tozeroy", fillcolor="rgba(163,113,247,0.15)",
            line=dict(color="#a371f7",width=2), name="Power kW",
        ))
    fig_pw.add_hline(y=m_info["rated_power_kw"],
                      line_dash="dash", line_color="#2ea043",
                      annotation_text=f"Nominal {m_info['rated_power_kw']} kW",
                      annotation_font_size=10)
    fig_pw.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark", height=260, margin=dict(l=0,r=0,t=10,b=0),
    )
    fig_pw.update_yaxes(ticksuffix=" kW", gridcolor="#21262d")
    fig_pw.update_xaxes(gridcolor="#21262d")
    st.plotly_chart(fig_pw, use_container_width=True)

with colb:
    st.markdown('<div class="section-title">📐 Power Factor</div>', unsafe_allow_html=True)
    fig_pf = go.Figure()
    if granularity == "Harian":
        pf_vals = df["avg_pf"]
        x_vals  = df["date"]
    else:
        pf_vals = df["power_factor"]
        x_vals  = df["timestamp"]

    pf_colors = pf_vals.apply(
        lambda x: "#2ea043" if x >= 0.85 else ("#d29922" if x >= 0.80 else "#f85149")
    ).tolist()
    fig_pf.add_trace(go.Scatter(
        x=x_vals, y=pf_vals, mode="lines",
        line=dict(color="#2ea043", width=2), name="PF",
        fill="tozeroy", fillcolor="rgba(46,160,67,0.1)",
    ))
    fig_pf.add_hline(y=0.85, line_dash="dash", line_color="#2ea043",
                      annotation_text="Target PF ≥ 0.85", annotation_font_size=10)
    fig_pf.add_hline(y=0.82, line_dash="dash", line_color="#d29922",
                      annotation_text="Warning PF < 0.82", annotation_font_size=10)
    fig_pf.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark", height=260, margin=dict(l=0,r=0,t=10,b=0),
        yaxis_range=[0.70, 1.02],
    )
    fig_pf.update_yaxes(gridcolor="#21262d")
    fig_pf.update_xaxes(gridcolor="#21262d")
    st.plotly_chart(fig_pf, use_container_width=True)

# ── Energi Kumulatif ─────────────────────────────────────
st.markdown('<div class="section-title">📈 Akumulasi Energi (kWh)</div>', unsafe_allow_html=True)

if granularity == "Harian":
    energy_cum = df[["date","total_energy_kwh"]].copy()
    energy_cum["cumulative_kwh"] = energy_cum["total_energy_kwh"].cumsum()
    fig_e = go.Figure()
    fig_e.add_trace(go.Scatter(
        x=energy_cum["date"], y=energy_cum["cumulative_kwh"],
        fill="tozeroy", fillcolor="rgba(240,180,41,0.10)",
        line=dict(color="#f0b429", width=2.5), name="Kumulatif kWh",
    ))
    fig_e.add_trace(go.Bar(
        x=energy_cum["date"], y=energy_cum["total_energy_kwh"],
        name="Harian kWh", marker_color="rgba(240,180,41,0.4)",
        yaxis="y2",
    ))
    fig_e.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark", height=280, margin=dict(l=0,r=0,t=10,b=0),
        yaxis=dict(title="Kumulatif kWh", gridcolor="#21262d"),
        yaxis2=dict(title="Harian kWh", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.05),
    )
    fig_e.update_xaxes(gridcolor="#21262d")
    st.plotly_chart(fig_e, use_container_width=True)

# ── Konsumsi per bulan (all machines type perbandingan) ──
st.markdown('<div class="section-title">📊 Perbandingan Energi Bulanan — Semua Tipe Mesin</div>',
            unsafe_allow_html=True)
energy_type = energy.groupby(["month","machine_type"])["energy_kwh"].sum().reset_index()
fig_em = px.bar(
    energy_type, x="month", y="energy_kwh", color="machine_type",
    labels={"energy_kwh":"kWh","month":"Bulan","machine_type":"Tipe"},
    template="plotly_dark",
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig_em.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=300, margin=dict(l=0,r=0,t=10,b=0),
    legend=dict(orientation="h", y=1.05),
)
fig_em.update_yaxes(ticksuffix=" kWh", gridcolor="#21262d")
fig_em.update_xaxes(gridcolor="#21262d", tickangle=-30)
st.plotly_chart(fig_em, use_container_width=True)

# ── Mesin terboros ────────────────────────────────────────
st.markdown('<div class="section-title">🏆 Ranking Mesin Terboros (Total kWh 2024)</div>',
            unsafe_allow_html=True)
top_energy = energy.groupby(["machine_id","machine_type"])["energy_kwh"].sum().reset_index()
top_energy = top_energy.sort_values("energy_kwh", ascending=False).head(15).reset_index(drop=True)
top_energy["total_kwh_fmt"] = top_energy["energy_kwh"].apply(lambda x: f"{x:,.0f}")
top_energy.index += 1
st.dataframe(
    top_energy[["machine_id","machine_type","total_kwh_fmt"]].rename(
        columns={"machine_id":"Mesin","machine_type":"Tipe","total_kwh_fmt":"Total kWh"}),
    use_container_width=True, hide_index=False, height=380,
)

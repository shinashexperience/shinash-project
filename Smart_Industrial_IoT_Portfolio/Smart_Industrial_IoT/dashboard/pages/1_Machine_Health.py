"""
Page 2 — Machine Health
Gauge health score, trend suhu & vibrasi per mesin
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os, numpy as np

st.set_page_config(page_title="Machine Health | IoT Dashboard", page_icon="🔧", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0d1117; }
[data-testid="stSidebar"] { background-color: #161b22; }
body, p, span, div { color: #e6edf3; }
.section-title {
    font-size:15px; font-weight:600; color:#58a6ff;
    border-left:3px solid #1f6feb; padding-left:10px; margin:18px 0 10px;
}
.metric-card {
    background:#161b22; border:1px solid #30363d; border-radius:10px;
    padding:14px 16px; text-align:center;
}
.metric-val { font-size:28px; font-weight:700; }
.metric-lbl { font-size:11px; color:#8b949e; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
RAW  = os.path.join(BASE, "data", "raw")

@st.cache_data(ttl=300)
def load():
    status = pd.read_csv(os.path.join(PROC, "machine_status.csv"))
    daily  = pd.read_csv(os.path.join(PROC, "sensor_daily.csv"))
    daily["date"] = pd.to_datetime(daily["date"])
    hourly = pd.read_csv(os.path.join(PROC, "sensor_hourly.csv"))
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])
    master = pd.read_csv(os.path.join(PROC, "machine_master.csv"))
    alarms = pd.read_csv(os.path.join(RAW,  "alarm_history.csv"), parse_dates=["timestamp"])
    return status, daily, hourly, master, alarms

status, daily, hourly, master, alarms = load()

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Machine Health")
    machine_ids = sorted(master["machine_id"].tolist())
    sel_machine = st.selectbox("Pilih Mesin", machine_ids, index=0)

    date_max = daily["date"].max().date()
    date_min = daily["date"].min().date()
    sel_start = st.date_input("Tanggal Mulai", value=date_max - pd.Timedelta(days=29),
                               min_value=date_min, max_value=date_max)
    sel_end   = st.date_input("Tanggal Akhir", value=date_max,
                               min_value=date_min, max_value=date_max)
    st.divider()
    st.page_link("Home.py", label="← Kembali ke Home")

d_start = pd.Timestamp(sel_start)
d_end   = pd.Timestamp(sel_end)

# ── Filter data ──────────────────────────────────────────
m_info   = master[master["machine_id"] == sel_machine].iloc[0]
m_status = status[status["machine_id"] == sel_machine].iloc[0]
daily_m  = daily[(daily["machine_id"] == sel_machine) &
                  (daily["date"] >= d_start) & (daily["date"] <= d_end)]
hourly_m = hourly[(hourly["machine_id"] == sel_machine) &
                   (hourly["timestamp"] >= d_start) & (hourly["timestamp"] <= d_end)]
alarms_m = alarms[(alarms["machine_id"] == sel_machine) &
                   (alarms["timestamp"] >= d_start) & (alarms["timestamp"] <= d_end)]

# ── Header ───────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,#1f2937,#111827);border-radius:12px;padding:20px 28px;margin-bottom:18px;border:1px solid #374151;">
  <h2 style="color:#e6edf3;margin:0;font-size:22px;">🔧 Machine Health — {sel_machine}</h2>
  <p style="color:#9ca3af;margin:4px 0 0;font-size:13px;">
    Tipe: <b style="color:#58a6ff">{m_info['machine_type']}</b> &nbsp;|&nbsp;
    Lokasi: {m_info['location']} &nbsp;|&nbsp;
    Kapasitas: {m_info['rated_power_kw']} kW &nbsp;|&nbsp;
    Tahun Pasang: {int(m_info['install_year'])} &nbsp;|&nbsp;
    Usia: <b style="color:#f0b429">{2024 - int(m_info['install_year'])} tahun</b>
  </p>
</div>
""", unsafe_allow_html=True)

# ── ROW 1: Gauge + Metrics ───────────────────────────────
col_gauge, col_m1, col_m2, col_m3, col_m4 = st.columns([2, 1, 1, 1, 1])

health = m_status["health_score"]
hs_color = "#2ea043" if health >= 75 else ("#d29922" if health >= 50 else "#f85149")
hs_label = "GOOD" if health >= 75 else ("WARNING" if health >= 50 else "CRITICAL")

with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health,
        delta={"reference": 80, "suffix": "%"},
        number={"suffix": "%", "font": {"size": 40, "color": hs_color}},
        title={"text": f"Health Score<br><b style='color:{hs_color}'>{hs_label}</b>",
               "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#6e7681"},
            "bar":  {"color": hs_color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0, 50],  "color": "rgba(248,81,73,0.15)"},
                {"range": [50, 75], "color": "rgba(210,153,34,0.15)"},
                {"range": [75, 100],"color": "rgba(46,160,67,0.15)"},
            ],
            "threshold": {"line": {"color": "white","width": 2}, "value": health},
        },
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
        height=240, margin=dict(l=20,r=20,t=30,b=10),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

def mcard(col, val, lbl, color="#e6edf3", suffix=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-val" style="color:{color}">{val}{suffix}</div>
        <div class="metric-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

avg_t  = daily_m["avg_temperature"].mean() if len(daily_m) else 0
max_t  = daily_m["max_temperature"].max()  if len(daily_m) else 0
avg_v  = daily_m["avg_vibration"].mean()   if len(daily_m) else 0
max_v  = daily_m["max_vibration"].max()    if len(daily_m) else 0

t_color = "#2ea043" if avg_t < 65 else ("#d29922" if avg_t < 70 else "#f85149")
v_color = "#2ea043" if avg_v < 2.8 else ("#d29922" if avg_v < 3.5 else "#f85149")

mcard(col_m1, f"{avg_t:.1f}",  "Rata-rata Suhu (°C)",    t_color)
mcard(col_m2, f"{max_t:.1f}",  "Suhu Maks (°C)",         "#f85149" if max_t > 80 else "#d29922")
mcard(col_m3, f"{avg_v:.3f}",  "Rata-rata Vibrasi (mm/s)",v_color)
mcard(col_m4, f"{len(alarms_m):,}", "Total Alarm",        "#d29922" if len(alarms_m) > 10 else "#2ea043")

st.markdown("")

# ── ROW 2: Temperature Trend + Vibration Trend ───────────
cola, colb = st.columns(2)

with cola:
    st.markdown('<div class="section-title">🌡️ Tren Suhu Harian</div>', unsafe_allow_html=True)
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(
        x=daily_m["date"], y=daily_m["avg_temperature"],
        name="Rata-rata", line=dict(color="#58a6ff", width=2), fill="tozeroy",
        fillcolor="rgba(88,166,255,0.08)",
    ))
    fig_t.add_trace(go.Scatter(
        x=daily_m["date"], y=daily_m["max_temperature"],
        name="Maks", line=dict(color="#f0b429", width=1.5, dash="dot"),
    ))
    fig_t.add_hline(y=70, line_dash="dash", line_color="#d29922",
                    annotation_text="Warning 70°C", annotation_font_size=10)
    fig_t.add_hline(y=80, line_dash="dash", line_color="#f85149",
                    annotation_text="Critical 80°C", annotation_font_size=10)
    fig_t.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark", height=260, margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", y=1.05),
    )
    fig_t.update_yaxes(ticksuffix=" °C", gridcolor="#21262d")
    fig_t.update_xaxes(gridcolor="#21262d")
    st.plotly_chart(fig_t, use_container_width=True)

with colb:
    st.markdown('<div class="section-title">📳 Tren Vibrasi Harian</div>', unsafe_allow_html=True)
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(
        x=daily_m["date"], y=daily_m["avg_vibration"],
        name="Rata-rata", line=dict(color="#a371f7", width=2), fill="tozeroy",
        fillcolor="rgba(163,113,247,0.08)",
    ))
    fig_v.add_trace(go.Scatter(
        x=daily_m["date"], y=daily_m["max_vibration"],
        name="Maks", line=dict(color="#f47067", width=1.5, dash="dot"),
    ))
    fig_v.add_hline(y=3.5, line_dash="dash", line_color="#d29922",
                    annotation_text="Warning 3.5 mm/s", annotation_font_size=10)
    fig_v.add_hline(y=4.5, line_dash="dash", line_color="#f85149",
                    annotation_text="Critical 4.5 mm/s", annotation_font_size=10)
    fig_v.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark", height=260, margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", y=1.05),
    )
    fig_v.update_yaxes(ticksuffix=" mm/s", gridcolor="#21262d")
    fig_v.update_xaxes(gridcolor="#21262d")
    st.plotly_chart(fig_v, use_container_width=True)

# ── ROW 3: Box plot distribusi suhu per jam ───────────────
st.markdown('<div class="section-title">🕐 Distribusi Suhu per Jam (Pola Beban Shift)</div>',
            unsafe_allow_html=True)
if len(hourly_m):
    hourly_m = hourly_m.copy()
    hourly_m["jam"] = hourly_m["timestamp"].dt.hour
    hourly_box = hourly_m.groupby("jam")["temperature"].apply(list).reset_index()

    fig_box = go.Figure()
    for _, r in hourly_box.iterrows():
        fig_box.add_trace(go.Box(
            y=r["temperature"], name=f"{int(r['jam']):02d}:00",
            marker_color="#1f6feb", showlegend=False, boxmean=True,
        ))
    fig_box.add_hline(y=70, line_dash="dash", line_color="#d29922", annotation_text="Warning")
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark", height=280, margin=dict(l=0,r=0,t=10,b=0),
    )
    fig_box.update_yaxes(ticksuffix=" °C", gridcolor="#21262d")
    fig_box.update_xaxes(gridcolor="#21262d", title="Jam")
    st.plotly_chart(fig_box, use_container_width=True)

# ── ROW 4: Alarm history untuk mesin ini ─────────────────
st.markdown('<div class="section-title">🚨 Riwayat Alarm Mesin Ini</div>', unsafe_allow_html=True)
if len(alarms_m):
    alarm_disp = alarms_m.sort_values("timestamp", ascending=False).head(50).copy()
    alarm_disp["Waktu"] = alarm_disp["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    alarm_disp["Severity"] = alarm_disp["severity"].apply(
        lambda s: "🔴 Critical" if s == "Critical" else "🟡 Warning")
    st.dataframe(
        alarm_disp[["Waktu","alarm_type","Severity","value"]].rename(
            columns={"alarm_type":"Tipe Alarm","value":"Nilai"}),
        use_container_width=True, hide_index=True, height=240,
    )
else:
    st.success("✅ Tidak ada alarm pada periode ini.")

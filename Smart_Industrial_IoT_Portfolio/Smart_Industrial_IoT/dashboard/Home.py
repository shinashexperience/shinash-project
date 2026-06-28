"""
Smart Industrial IoT Monitoring Platform
PT Nusantara Steel Manufacturing
─────────────────────────────────────────
Home.py  —  Executive Summary
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

# ─────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Industrial IoT | PT Nusantara Steel",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Font & Background ── */
[data-testid="stAppViewContainer"] { background-color: #0d1117; }
[data-testid="stSidebar"] { background-color: #161b22; }
body, p, span, div { color: #e6edf3; }

/* ── Header Banner ── */
.banner {
    background: linear-gradient(135deg, #1f6feb 0%, #0d419d 50%, #0a2d6d 100%);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
}
.banner h1 { color: #ffffff; font-size: 26px; margin: 0; }
.banner p  { color: #a5d6ff; font-size: 13px; margin: 4px 0 0; }

/* ── KPI Cards ── */
.kpi-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
}
.kpi-value { font-size: 36px; font-weight: 700; }
.kpi-label { font-size: 12px; color: #8b949e; margin-top: 4px; }
.kpi-green  { color: #2ea043; }
.kpi-red    { color: #f85149; }
.kpi-yellow { color: #d29922; }
.kpi-blue   { color: #1f6feb; }
.kpi-white  { color: #e6edf3; }

/* ── Section title ── */
.section-title {
    font-size: 16px;
    font-weight: 600;
    color: #58a6ff;
    border-left: 3px solid #1f6feb;
    padding-left: 10px;
    margin: 24px 0 12px;
}

/* ── Alarm badges ── */
.badge-critical {
    background: #3d1a1a; color: #f85149;
    border: 1px solid #f85149;
    border-radius: 6px; padding: 2px 8px; font-size: 11px;
}
.badge-warning {
    background: #2e2007; color: #e3b341;
    border: 1px solid #e3b341;
    border-radius: 6px; padding: 2px 8px; font-size: 11px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC  = os.path.join(BASE, "data", "processed")
RAW   = os.path.join(BASE, "data", "raw")

@st.cache_data(ttl=300)
def load_data():
    status   = pd.read_csv(os.path.join(PROC, "machine_status.csv"))
    daily    = pd.read_csv(os.path.join(PROC, "sensor_daily.csv"))
    daily["date"] = pd.to_datetime(daily["date"])
    alarms   = pd.read_csv(os.path.join(RAW,  "alarm_history.csv"), parse_dates=["timestamp"])
    master   = pd.read_csv(os.path.join(PROC, "machine_master.csv"))
    energy_m = pd.read_csv(os.path.join(PROC, "energy_monthly.csv"))
    return status, daily, alarms, master, energy_m

status, daily, alarms, master, energy_m = load_data()

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/factory.png", width=60)
    st.markdown("### 🏭 PT Nusantara Steel")
    st.markdown("**Industrial IoT Platform**")
    st.divider()

    machine_types = ["Semua"] + sorted(master["machine_type"].unique().tolist())
    sel_type = st.selectbox("Filter Tipe Mesin", machine_types)

    date_min = daily["date"].min().date()
    date_max = daily["date"].max().date()
    sel_date = st.date_input(
        "Rentang Tanggal",
        value=(date_max - pd.Timedelta(days=29), date_max),
        min_value=date_min, max_value=date_max,
    )
    st.divider()
    st.markdown("**Navigasi:**")
    st.page_link("pages/1_Machine_Health.py",  label="🔧 Machine Health",  icon="🔧")
    st.page_link("pages/2_Power_Monitor.py",   label="⚡ Power Monitor",   icon="⚡")
    st.page_link("pages/3_Alarm_Center.py",    label="🚨 Alarm Center",    icon="🚨")
    st.page_link("pages/4_Maintenance.py",     label="🛠️ Maintenance",     icon="🛠️")
    st.page_link("pages/5_Analytics.py",       label="📊 Analytics",       icon="📊")
    st.divider()
    st.caption("Data: Simulasi 2024 | Interval: 10 mnt")

# ─────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <h1>🏭 Smart Industrial IoT Monitoring Platform</h1>
  <p>PT Nusantara Steel Manufacturing  ·  Executive Summary Dashboard  ·  Periode: Januari – Desember 2024</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# FILTER TANGGAL
# ─────────────────────────────────────────────────────────
try:
    d_start, d_end = pd.Timestamp(sel_date[0]), pd.Timestamp(sel_date[1])
except:
    d_end   = pd.Timestamp(daily["date"].max())
    d_start = d_end - pd.Timedelta(days=29)

daily_f = daily[(daily["date"] >= d_start) & (daily["date"] <= d_end)]
if sel_type != "Semua":
    machines_sel = master[master["machine_type"] == sel_type]["machine_id"].tolist()
    daily_f = daily_f[daily_f["machine_id"].isin(machines_sel)]
    status_f = status[status["machine_type"] == sel_type]
else:
    status_f = status

# Alarms filter
alarms_f = alarms[(alarms["timestamp"] >= d_start) & (alarms["timestamp"] <= d_end)]
if sel_type != "Semua":
    alarms_f = alarms_f[alarms_f["machine_id"].isin(machines_sel)]

# ─────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────
total   = len(status_f)
online  = len(status_f[status_f["status"].isin(["Good","Warning"])])
offline = len(status_f[status_f["status"] == "Offline"])
critical_m = len(status_f[status_f["status"] == "Critical"])
active_alarms = len(alarms_f[alarms_f["severity"] == "Critical"])
daily_energy  = daily_f.groupby("date")["total_energy_kwh"].sum()
avg_daily_energy = daily_energy.mean() if len(daily_energy) else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)

def kpi_card(col, value, label, color, suffix=""):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value {color}">{value}{suffix}</div>
        <div class="kpi-label">{label}</div>
    </div>""", unsafe_allow_html=True)

kpi_card(c1, total,     "Total Mesin",      "kpi-white")
kpi_card(c2, online,    "Online",           "kpi-green")
kpi_card(c3, offline,   "Offline",          "kpi-red")
kpi_card(c4, critical_m,"Critical",         "kpi-red")
kpi_card(c5, active_alarms, "Active Alarms","kpi-yellow")
kpi_card(c6, f"{avg_daily_energy:,.0f}", "Avg Daily kWh", "kpi-blue")

st.markdown("")

# ─────────────────────────────────────────────────────────
# ROW 2: Machine Status + Alarm Severity
# ─────────────────────────────────────────────────────────
col_l, col_r = st.columns([2, 1])

with col_l:
    st.markdown('<div class="section-title">📊 Status Mesin per Tipe</div>', unsafe_allow_html=True)
    status_agg = status_f.groupby(["machine_type","status"]).size().reset_index(name="count")
    color_map = {"Good":"#2ea043","Warning":"#d29922","Critical":"#f85149","Offline":"#6e7681"}
    fig = px.bar(
        status_agg, x="machine_type", y="count", color="status",
        color_discrete_map=color_map,
        labels={"machine_type":"Tipe Mesin","count":"Jumlah","status":"Status"},
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=0,r=0,t=30,b=0), height=280,
    )
    fig.update_xaxes(gridcolor="#21262d")
    fig.update_yaxes(gridcolor="#21262d")
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.markdown('<div class="section-title">🚨 Distribusi Alarm</div>', unsafe_allow_html=True)
    sev_counts = alarms_f["severity"].value_counts().reset_index()
    sev_counts.columns = ["severity", "count"]
    fig2 = go.Figure(go.Pie(
        labels=sev_counts["severity"], values=sev_counts["count"],
        hole=0.55,
        marker_colors=["#f85149","#d29922"],
        textinfo="percent+label",
        textfont_size=12,
    ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, margin=dict(l=0,r=0,t=30,b=0), height=280,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────
# ROW 3: Energy Trend
# ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">⚡ Tren Konsumsi Energi Harian (kWh)</div>', unsafe_allow_html=True)
energy_trend = daily_f.groupby("date")["total_energy_kwh"].sum().reset_index()
energy_trend.columns = ["date","total_kwh"]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=energy_trend["date"], y=energy_trend["total_kwh"],
    fill="tozeroy", fillcolor="rgba(31,111,235,0.15)",
    line=dict(color="#1f6feb", width=2),
    name="Total kWh",
))
# Moving average
if len(energy_trend) >= 7:
    ma7 = energy_trend["total_kwh"].rolling(7).mean()
    fig3.add_trace(go.Scatter(
        x=energy_trend["date"], y=ma7,
        line=dict(color="#e3b341", width=2, dash="dash"),
        name="MA 7 Hari",
    ))
fig3.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    template="plotly_dark", height=260,
    legend=dict(orientation="h", yanchor="bottom", y=1.0),
    margin=dict(l=0,r=0,t=10,b=0),
)
fig3.update_xaxes(gridcolor="#21262d", showgrid=True)
fig3.update_yaxes(gridcolor="#21262d", ticksuffix=" kWh")
st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────
# ROW 4: Top Machines (worst health) + Recent Alarms
# ─────────────────────────────────────────────────────────
col_a, col_b = st.columns([1, 1])

with col_a:
    st.markdown('<div class="section-title">⚠️ Mesin Health Score Terburuk</div>', unsafe_allow_html=True)
    worst = status_f.nsmallest(8, "health_score")[
        ["machine_id","machine_type","health_score","status"]
    ].reset_index(drop=True)

    def color_score(val):
        if val >= 75: return "🟢"
        if val >= 50: return "🟡"
        return "🔴"

    worst["🔔"] = worst["health_score"].apply(color_score)
    worst["Health Score"] = worst["health_score"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(
        worst[["🔔","machine_id","machine_type","Health Score","status"]].rename(
            columns={"machine_id":"Mesin","machine_type":"Tipe","status":"Status"}),
        use_container_width=True, hide_index=True, height=300,
    )

with col_b:
    st.markdown('<div class="section-title">🚨 Alarm Terbaru</div>', unsafe_allow_html=True)
    recent_alarms = alarms_f.sort_values("timestamp", ascending=False).head(20).reset_index(drop=True)
    recent_alarms["Waktu"] = recent_alarms["timestamp"].dt.strftime("%m-%d %H:%M")
    recent_alarms["Sev"]   = recent_alarms["severity"].apply(
        lambda s: "🔴 Critical" if s=="Critical" else "🟡 Warning")
    st.dataframe(
        recent_alarms[["Waktu","machine_id","alarm_type","Sev","value"]].rename(
            columns={"machine_id":"Mesin","alarm_type":"Tipe Alarm","Sev":"Severity","value":"Nilai"}),
        use_container_width=True, hide_index=True, height=300,
    )

# ─────────────────────────────────────────────────────────
# ROW 5: Health Score Bar Chart
# ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🏆 Health Score Semua Mesin</div>', unsafe_allow_html=True)
hs = status_f.sort_values("health_score").copy()
colors = hs["health_score"].apply(
    lambda x: "#f85149" if x < 50 else ("#d29922" if x < 75 else "#2ea043")
).tolist()
fig4 = go.Figure(go.Bar(
    x=hs["machine_id"], y=hs["health_score"],
    marker_color=colors,
    text=hs["health_score"].apply(lambda x: f"{x:.0f}%"),
    textposition="outside",
))
fig4.add_hline(y=75, line_dash="dash", line_color="#2ea043", annotation_text="Good ≥75%")
fig4.add_hline(y=50, line_dash="dash", line_color="#d29922", annotation_text="Warning ≥50%")
fig4.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    template="plotly_dark", height=320,
    margin=dict(l=0,r=0,t=10,b=60), yaxis_range=[0, 115],
)
fig4.update_xaxes(gridcolor="#21262d", tickangle=-45)
fig4.update_yaxes(gridcolor="#21262d", ticksuffix="%")
st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#6e7681; font-size:12px;">
  Smart Industrial IoT Platform  ·  PT Nusantara Steel Manufacturing (Simulasi)  ·
  Data: 1,897,344 records  ·  Dibuat dengan Python, Streamlit & Plotly
</div>
""", unsafe_allow_html=True)

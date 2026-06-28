"""
Page 5 — Maintenance
Upcoming Maintenance, MTBF, MTTR, Downtime, Cost Analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
from datetime import datetime

st.set_page_config(page_title="Maintenance | IoT Dashboard", page_icon="🛠️", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0d1117; }
[data-testid="stSidebar"]          { background-color: #161b22; }
body, p, span, div { color: #e6edf3; }
.section-title {
    font-size:15px; font-weight:600; color:#58a6ff;
    border-left:3px solid #2ea043; padding-left:10px; margin:18px 0 10px;
}
.kpi { background:#161b22; border:1px solid #30363d; border-radius:10px;
       padding:14px 16px; text-align:center; }
.kv  { font-size:28px; font-weight:700; }
.kl  { font-size:11px; color:#8b949e; margin-top:3px; }
.upcoming-card {
    background:#161b22; border:1px solid #30363d; border-radius:8px;
    padding:12px 16px; margin-bottom:8px;
}
</style>
""", unsafe_allow_html=True)

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW   = os.path.join(BASE, "data", "raw")
PROC  = os.path.join(BASE, "data", "processed")

@st.cache_data(ttl=300)
def load():
    maint  = pd.read_csv(os.path.join(RAW,  "maintenance_history.csv"))
    maint["date"] = pd.to_datetime(maint["date"])
    master = pd.read_csv(os.path.join(PROC, "machine_master.csv"))
    status = pd.read_csv(os.path.join(PROC, "machine_status.csv"))
    alarms = pd.read_csv(os.path.join(RAW,  "alarm_history.csv"), parse_dates=["timestamp"])
    return maint, master, status, alarms

maint, master, status, alarms = load()

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛠️ Maintenance")
    machine_types = ["Semua"] + sorted(master["machine_type"].unique().tolist())
    sel_type = st.selectbox("Filter Tipe Mesin", machine_types)
    maint_filter = ["Semua","Preventive Maintenance","Corrective Maintenance"]
    sel_mtype = st.selectbox("Filter Tipe Maintenance", maint_filter)
    st.divider()
    st.page_link("Home.py", label="← Kembali ke Home")

# ── Filter ───────────────────────────────────────────────
maint_f = maint.copy()
if sel_type != "Semua":
    ids_sel = master[master["machine_type"] == sel_type]["machine_id"].tolist()
    maint_f = maint_f[maint_f["machine_id"].isin(ids_sel)]
if sel_mtype == "Preventive Maintenance":
    maint_f = maint_f[maint_f["type"] == "Preventive Maintenance"]
elif sel_mtype == "Corrective Maintenance":
    maint_f = maint_f[maint_f["type"] != "Preventive Maintenance"]

# ── Header ───────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0f2315,#071a0c);border-radius:12px;
     padding:20px 28px;margin-bottom:18px;border:1px solid #1a4025;">
  <h2 style="color:#e6edf3;margin:0;font-size:22px;">🛠️ Maintenance Management</h2>
  <p style="color:#9ca3af;margin:4px 0 0;font-size:13px;">
    MTBF · MTTR · Downtime · Cost Analysis — PT Nusantara Steel Manufacturing
  </p>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ──────────────────────────────────────────────
total_events  = len(maint_f)
preventive    = len(maint_f[maint_f["type"] == "Preventive Maintenance"])
corrective    = total_events - preventive
total_hours   = maint_f["duration_hours"].sum()
total_cost    = maint_f["cost_idr"].sum()
avg_duration  = maint_f["duration_hours"].mean() if total_events else 0

# MTBF & MTTR sederhana
# MTBF = total jam operasi / jumlah failure (corrective events)
total_op_hours = 365 * 24
n_failures     = corrective if corrective > 0 else 1
n_machines     = maint_f["machine_id"].nunique() if maint_f["machine_id"].nunique() > 0 else 1
mtbf_hours     = (total_op_hours * n_machines) / n_failures
mttr_hours     = maint_f[maint_f["type"] != "Preventive Maintenance"]["duration_hours"].mean() \
                 if corrective > 0 else 0

k1,k2,k3,k4,k5,k6 = st.columns(6)
def kpi(col, v, lbl, color="#e6edf3"):
    col.markdown(f'<div class="kpi"><div class="kv" style="color:{color}">{v}</div>'
                 f'<div class="kl">{lbl}</div></div>', unsafe_allow_html=True)

kpi(k1, f"{total_events}",         "Total Event",            "#e6edf3")
kpi(k2, f"{preventive}",           "Preventive",             "#2ea043")
kpi(k3, f"{corrective}",           "Corrective",             "#f85149")
kpi(k4, f"{mtbf_hours:,.0f} jam",  "MTBF",                   "#58a6ff")
kpi(k5, f"{mttr_hours:.1f} jam",   "MTTR",                   "#d29922")
kpi(k6, f"Rp {total_cost/1e6:.1f}M","Total Biaya",           "#a371f7")

st.markdown("")

# ── ROW 2: Upcoming Maintenance (simulasi 30 hari ke depan) + Tipe Breakdown
col_l, col_r = st.columns([2, 1])

with col_l:
    st.markdown('<div class="section-title">📅 Jadwal Maintenance Mendatang (Simulasi)</div>',
                unsafe_allow_html=True)
    # Simulasi jadwal next PM — 90 hari setelah PM terakhir setiap mesin
    last_pm = (maint[maint["type"] == "Preventive Maintenance"]
               .groupby("machine_id")["date"].max().reset_index())
    last_pm["next_due"] = last_pm["date"] + pd.Timedelta(days=90)
    last_pm = last_pm.merge(master[["machine_id","machine_type","location"]], on="machine_id")
    last_pm["days_left"] = (last_pm["next_due"] - pd.Timestamp("2024-12-31")).dt.days
    upcoming = last_pm.sort_values("next_due").head(12).reset_index(drop=True)
    upcoming.index += 1

    def urgency(d):
        if d <= 0:    return "🔴 Terlambat"
        if d <= 7:    return "🟡 Segera"
        if d <= 30:   return "🟠 Bulan Ini"
        return "🟢 On Schedule"

    upcoming["Status"] = upcoming["days_left"].apply(urgency)
    upcoming["next_due_str"] = upcoming["next_due"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        upcoming[["machine_id","machine_type","location","next_due_str","Status"]].rename(
            columns={"machine_id":"Mesin","machine_type":"Tipe",
                     "location":"Lokasi","next_due_str":"Jadwal Berikutnya"}),
        use_container_width=True, hide_index=False, height=380,
    )

with col_r:
    st.markdown('<div class="section-title">📊 Breakdown Tipe</div>', unsafe_allow_html=True)
    type_counts = maint_f["type"].apply(
        lambda x: "Preventive" if x == "Preventive Maintenance" else "Corrective"
    ).value_counts().reset_index()
    type_counts.columns = ["tipe","count"]
    fig_pie = go.Figure(go.Pie(
        labels=type_counts["tipe"], values=type_counts["count"],
        hole=0.55, textinfo="percent+label",
        marker_colors=["#2ea043","#f85149"],
        textfont_size=12,
    ))
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", height=200,
        margin=dict(l=0,r=0,t=10,b=0), showlegend=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:8px">🔩 Komponen Sering Diganti</div>',
                unsafe_allow_html=True)
    comp_counts = (maint_f[maint_f["type"] != "Preventive Maintenance"]
                   ["component"].value_counts().head(6).reset_index())
    comp_counts.columns = ["Komponen","Frekuensi"]
    st.dataframe(comp_counts, use_container_width=True, hide_index=True, height=200)

# ── ROW 3: MTBF per Mesin ────────────────────────────────
st.markdown('<div class="section-title">📐 MTBF per Mesin (jam operasi / jumlah failure)</div>',
            unsafe_allow_html=True)
corr_per_machine = (maint_f[maint_f["type"] != "Preventive Maintenance"]
                    .groupby("machine_id").size().reset_index(name="n_failures"))
corr_per_machine["mtbf"] = total_op_hours / corr_per_machine["n_failures"].replace(0, 1)
corr_per_machine = corr_per_machine.merge(master[["machine_id","machine_type"]], on="machine_id")
corr_per_machine = corr_per_machine.sort_values("mtbf", ascending=False)

fig_mtbf = go.Figure(go.Bar(
    x=corr_per_machine["machine_id"], y=corr_per_machine["mtbf"],
    marker_color=[
        "#2ea043" if v >= 1500 else ("#d29922" if v >= 800 else "#f85149")
        for v in corr_per_machine["mtbf"]
    ],
    text=corr_per_machine["mtbf"].apply(lambda x: f"{x:,.0f}h"),
    textposition="outside",
))
fig_mtbf.add_hline(y=1500, line_dash="dash", line_color="#2ea043",
                    annotation_text="Target MTBF ≥ 1500h")
fig_mtbf.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    template="plotly_dark", height=300, margin=dict(l=0,r=0,t=10,b=60),
    yaxis_title="MTBF (jam)",
)
fig_mtbf.update_xaxes(gridcolor="#21262d", tickangle=-40)
fig_mtbf.update_yaxes(gridcolor="#21262d")
st.plotly_chart(fig_mtbf, use_container_width=True)

# ── ROW 4: Biaya Maintenance per Tipe per Bulan ───────────
cola, colb = st.columns(2)

with cola:
    st.markdown('<div class="section-title">💰 Biaya Maintenance Bulanan</div>',
                unsafe_allow_html=True)
    maint_f2 = maint_f.copy()
    maint_f2["bulan"] = maint_f2["date"].dt.to_period("M").astype(str)
    maint_f2["tipe2"] = maint_f2["type"].apply(
        lambda x: "Preventive" if x == "Preventive Maintenance" else "Corrective")
    cost_monthly = maint_f2.groupby(["bulan","tipe2"])["cost_idr"].sum().reset_index()
    fig_cost = px.bar(
        cost_monthly, x="bulan", y="cost_idr", color="tipe2",
        color_discrete_map={"Preventive":"#2ea043","Corrective":"#f85149"},
        labels={"cost_idr":"Biaya (Rp)","bulan":"Bulan","tipe2":"Tipe"},
        template="plotly_dark",
    )
    fig_cost.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=280, margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", y=1.05),
    )
    fig_cost.update_yaxes(tickprefix="Rp ", gridcolor="#21262d")
    fig_cost.update_xaxes(gridcolor="#21262d", tickangle=-30)
    st.plotly_chart(fig_cost, use_container_width=True)

with colb:
    st.markdown('<div class="section-title">⏱️ Durasi Maintenance per Tipe Mesin</div>',
                unsafe_allow_html=True)
    dur_box = maint_f.merge(master[["machine_id","machine_type"]], on="machine_id")
    fig_dur = px.box(
        dur_box, x="machine_type", y="duration_hours", color="machine_type",
        template="plotly_dark",
        labels={"duration_hours":"Durasi (jam)","machine_type":"Tipe Mesin"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_dur.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=280, margin=dict(l=0,r=0,t=10,b=0), showlegend=False,
    )
    fig_dur.update_yaxes(gridcolor="#21262d", ticksuffix=" jam")
    fig_dur.update_xaxes(gridcolor="#21262d")
    st.plotly_chart(fig_dur, use_container_width=True)

# ── ROW 5: Downtime Ranking ───────────────────────────────
st.markdown('<div class="section-title">📉 Ranking Downtime per Mesin (Total Jam)</div>',
            unsafe_allow_html=True)
downtime = (maint_f.groupby("machine_id")["duration_hours"]
            .sum().reset_index().sort_values("duration_hours",ascending=False)
            .merge(master[["machine_id","machine_type"]], on="machine_id").head(15))
downtime["availability"] = ((total_op_hours - downtime["duration_hours"]) / total_op_hours * 100).clip(0,100)

fig_dt = go.Figure()
fig_dt.add_trace(go.Bar(
    x=downtime["machine_id"], y=downtime["duration_hours"],
    name="Downtime (jam)", marker_color="#f85149",
    yaxis="y",
))
fig_dt.add_trace(go.Scatter(
    x=downtime["machine_id"], y=downtime["availability"],
    name="Availability %", line=dict(color="#2ea043",width=2),
    mode="lines+markers", yaxis="y2",
))
fig_dt.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    template="plotly_dark", height=300, margin=dict(l=0,r=0,t=10,b=60),
    legend=dict(orientation="h", y=1.05),
    yaxis=dict(title="Downtime (jam)", gridcolor="#21262d"),
    yaxis2=dict(title="Availability %", overlaying="y", side="right",
                ticksuffix="%", range=[90,100]),
)
fig_dt.update_xaxes(gridcolor="#21262d", tickangle=-35)
st.plotly_chart(fig_dt, use_container_width=True)

# ── Tabel Maintenance History ─────────────────────────────
st.markdown('<div class="section-title">📋 Riwayat Maintenance</div>', unsafe_allow_html=True)
maint_show = maint_f.sort_values("date", ascending=False).copy()
maint_show["date_str"] = maint_show["date"].dt.strftime("%Y-%m-%d")
maint_show["cost_fmt"] = maint_show["cost_idr"].apply(lambda x: f"Rp {x:,}")
st.dataframe(
    maint_show[["date_str","machine_id","type","component","technician",
                "duration_hours","cost_fmt"]].rename(columns={
        "date_str":"Tanggal","machine_id":"Mesin","type":"Tipe",
        "component":"Komponen","technician":"Teknisi",
        "duration_hours":"Durasi (jam)","cost_fmt":"Biaya",
    }),
    use_container_width=True, hide_index=True, height=360,
)

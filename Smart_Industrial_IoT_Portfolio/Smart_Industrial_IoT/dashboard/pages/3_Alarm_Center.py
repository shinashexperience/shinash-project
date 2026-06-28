"""
Page 4 — Alarm Center
Critical Alarm, Warning Alarm, Alarm History, Heatmap
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

st.set_page_config(page_title="Alarm Center | IoT Dashboard", page_icon="🚨", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0d1117; }
[data-testid="stSidebar"]          { background-color: #161b22; }
body, p, span, div { color: #e6edf3; }
.section-title {
    font-size:15px; font-weight:600; color:#58a6ff;
    border-left:3px solid #f85149; padding-left:10px; margin:18px 0 10px;
}
.kpi { background:#161b22; border:1px solid #30363d; border-radius:10px;
       padding:14px 16px; text-align:center; }
.kv  { font-size:32px; font-weight:700; }
.kl  { font-size:11px; color:#8b949e; margin-top:3px; }
</style>
""", unsafe_allow_html=True)

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW    = os.path.join(BASE, "data", "raw")
PROC   = os.path.join(BASE, "data", "processed")

@st.cache_data(ttl=300)
def load():
    alarms = pd.read_csv(os.path.join(RAW,  "alarm_history.csv"), parse_dates=["timestamp"])
    master = pd.read_csv(os.path.join(PROC, "machine_master.csv"))
    status = pd.read_csv(os.path.join(PROC, "machine_status.csv"))
    return alarms, master, status

alarms, master, status = load()

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚨 Alarm Center")

    machine_types = ["Semua"] + sorted(master["machine_type"].unique().tolist())
    sel_type = st.selectbox("Filter Tipe Mesin", machine_types)

    alarm_types = ["Semua"] + sorted(alarms["alarm_type"].unique().tolist())
    sel_alarm_type = st.selectbox("Filter Tipe Alarm", alarm_types)

    severity_opts = ["Semua", "Critical", "Warning"]
    sel_sev = st.selectbox("Filter Severity", severity_opts)

    date_max = alarms["timestamp"].max().date()
    date_min = alarms["timestamp"].min().date()
    sel_start = st.date_input("Mulai", date_max - pd.Timedelta(days=29),
                               min_value=date_min, max_value=date_max)
    sel_end   = st.date_input("Akhir", date_max,
                               min_value=date_min, max_value=date_max)
    st.divider()
    st.page_link("Home.py", label="← Kembali ke Home")

d_start = pd.Timestamp(sel_start)
d_end   = pd.Timestamp(sel_end)

# ── Apply Filter ─────────────────────────────────────────
df = alarms[(alarms["timestamp"] >= d_start) & (alarms["timestamp"] <= d_end)].copy()
if sel_type != "Semua":
    ids_sel = master[master["machine_type"] == sel_type]["machine_id"].tolist()
    df = df[df["machine_id"].isin(ids_sel)]
if sel_alarm_type != "Semua":
    df = df[df["alarm_type"] == sel_alarm_type]
if sel_sev != "Semua":
    df = df[df["severity"] == sel_sev]

# ── Header ───────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#2d1010,#1a0808);border-radius:12px;
     padding:20px 28px;margin-bottom:18px;border:1px solid #4a1515;">
  <h2 style="color:#e6edf3;margin:0;font-size:22px;">🚨 Alarm Center</h2>
  <p style="color:#9ca3af;margin:4px 0 0;font-size:13px;">
    Real-time alarm monitoring — PT Nusantara Steel Manufacturing
  </p>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ──────────────────────────────────────────────
total_alarms    = len(df)
critical_alarms = len(df[df["severity"] == "Critical"])
warning_alarms  = len(df[df["severity"] == "Warning"])
machines_alarm  = df["machine_id"].nunique()
worst_machine   = df["machine_id"].value_counts().index[0] if total_alarms else "-"
worst_type      = df["alarm_type"].value_counts().index[0] if total_alarms else "-"

k1,k2,k3,k4,k5,k6 = st.columns(6)
def kpi(col, v, lbl, color="#e6edf3"):
    col.markdown(f'<div class="kpi"><div class="kv" style="color:{color}">{v}</div>'
                 f'<div class="kl">{lbl}</div></div>', unsafe_allow_html=True)

kpi(k1, f"{total_alarms:,}",    "Total Alarm",         "#e6edf3")
kpi(k2, f"{critical_alarms:,}", "Critical",            "#f85149")
kpi(k3, f"{warning_alarms:,}",  "Warning",             "#d29922")
kpi(k4, f"{machines_alarm}",    "Mesin Terdampak",     "#58a6ff")
kpi(k5, worst_machine,           "Mesin Paling Banyak", "#f85149")
kpi(k6, worst_type[:18] if worst_type != "-" else "-", "Tipe Alarm Terbanyak", "#d29922")

st.markdown("")

# ── ROW 2: Trend Alarm Harian + Distribusi Tipe ──────────
col_l, col_r = st.columns([3, 2])

with col_l:
    st.markdown('<div class="section-title">📈 Tren Alarm Harian</div>', unsafe_allow_html=True)
    df_trend = df.copy()
    df_trend["date"] = df_trend["timestamp"].dt.date
    trend = df_trend.groupby(["date","severity"]).size().reset_index(name="count")

    fig_t = go.Figure()
    colors = {"Critical":"#f85149","Warning":"#d29922"}
    for sev, grp in trend.groupby("severity"):
        fig_t.add_trace(go.Bar(
            x=grp["date"], y=grp["count"],
            name=sev, marker_color=colors.get(sev,"#58a6ff"),
        ))
    fig_t.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark", height=280, margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", y=1.05),
    )
    fig_t.update_xaxes(gridcolor="#21262d")
    fig_t.update_yaxes(gridcolor="#21262d", title="Jumlah Alarm")
    st.plotly_chart(fig_t, use_container_width=True)

with col_r:
    st.markdown('<div class="section-title">🍩 Distribusi Tipe Alarm</div>', unsafe_allow_html=True)
    type_counts = df["alarm_type"].value_counts().reset_index()
    type_counts.columns = ["alarm_type", "count"]
    fig_d = go.Figure(go.Pie(
        labels=type_counts["alarm_type"], values=type_counts["count"],
        hole=0.5, textinfo="percent",
        marker=dict(colors=px.colors.qualitative.Bold),
        textfont_size=11,
    ))
    fig_d.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark", height=280, margin=dict(l=0,r=0,t=10,b=40),
        legend=dict(font_size=10, x=0, y=-0.2, orientation="h"),
    )
    st.plotly_chart(fig_d, use_container_width=True)

# ── ROW 3: Top 15 Machines by Alarm Count ────────────────
st.markdown('<div class="section-title">🏭 Mesin dengan Alarm Terbanyak</div>', unsafe_allow_html=True)
top_machines = (
    df.groupby(["machine_id","severity"]).size()
    .reset_index(name="count")
    .pivot_table(index="machine_id", columns="severity", values="count", fill_value=0)
    .reset_index()
)
if "Critical" not in top_machines.columns: top_machines["Critical"] = 0
if "Warning"  not in top_machines.columns: top_machines["Warning"]  = 0
top_machines["Total"] = top_machines.get("Critical",0) + top_machines.get("Warning",0)
top_machines = top_machines.merge(master[["machine_id","machine_type"]], on="machine_id")
top_machines = top_machines.sort_values("Total", ascending=False).head(15)

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=top_machines["machine_id"], y=top_machines.get("Critical", [0]*len(top_machines)),
    name="Critical", marker_color="#f85149",
))
fig_bar.add_trace(go.Bar(
    x=top_machines["machine_id"], y=top_machines.get("Warning", [0]*len(top_machines)),
    name="Warning", marker_color="#d29922",
))
fig_bar.update_layout(
    barmode="stack",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    template="plotly_dark", height=300, margin=dict(l=0,r=0,t=10,b=60),
    legend=dict(orientation="h", y=1.05),
)
fig_bar.update_xaxes(gridcolor="#21262d", tickangle=-35)
fig_bar.update_yaxes(gridcolor="#21262d", title="Jumlah Alarm")
st.plotly_chart(fig_bar, use_container_width=True)

# ── ROW 4: Alarm Heatmap (Jam vs Hari) ───────────────────
st.markdown('<div class="section-title">🗓️ Heatmap Alarm — Jam vs Hari dalam Seminggu</div>',
            unsafe_allow_html=True)
df["jam"]   = df["timestamp"].dt.hour
df["hari"]  = df["timestamp"].dt.day_name()
HARI_ORDER  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
HARI_ID     = ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"]
heatmap_data = df.groupby(["hari","jam"]).size().unstack(fill_value=0)
heatmap_data = heatmap_data.reindex(HARI_ORDER, fill_value=0)

fig_h = go.Figure(go.Heatmap(
    z=heatmap_data.values,
    x=[f"{h:02d}:00" for h in heatmap_data.columns],
    y=HARI_ID,
    colorscale=[[0,"#0d1117"],[0.3,"#3d1a1a"],[0.7,"#7b1414"],[1.0,"#f85149"]],
    showscale=True,
    colorbar=dict(title="Jml Alarm", tickfont=dict(color="#e6edf3")),
))
fig_h.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=280, margin=dict(l=0,r=0,t=10,b=0),
    font_color="#e6edf3",
    xaxis=dict(title="Jam"),
    yaxis=dict(title=""),
)
st.plotly_chart(fig_h, use_container_width=True)

# ── ROW 5: Tabel Alarm Terbaru ────────────────────────────
st.markdown('<div class="section-title">📋 Riwayat Alarm Terbaru</div>', unsafe_allow_html=True)
col_filter1, col_filter2 = st.columns([2,1])
with col_filter1:
    search = st.text_input("🔍 Cari mesin / tipe alarm...", placeholder="Contoh: MTR-001 atau High Temperature")
with col_filter2:
    n_show = st.slider("Tampilkan N baris", 20, 200, 50, step=10)

display = df.sort_values("timestamp", ascending=False).copy()
if search:
    mask = (display["machine_id"].str.contains(search, case=False, na=False) |
            display["alarm_type"].str.contains(search, case=False, na=False))
    display = display[mask]

display = display.head(n_show).reset_index(drop=True)
display["Waktu"]    = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
display["Severity"] = display["severity"].apply(
    lambda s: "🔴 Critical" if s=="Critical" else "🟡 Warning")

st.dataframe(
    display[["Waktu","machine_id","alarm_type","Severity","value"]].rename(
        columns={"machine_id":"Mesin","alarm_type":"Tipe Alarm","value":"Nilai"}),
    use_container_width=True, hide_index=True, height=380,
)

# ── Ekspor ───────────────────────────────────────────────
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Alarm History (CSV)",
    data=csv_data,
    file_name="alarm_history_filtered.csv",
    mime="text/csv",
)

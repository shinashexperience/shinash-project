"""
Page 6 — Analytics
Correlation · Heatmap · Anomaly Detection · Failure Prediction · Energy Forecast
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Analytics | IoT Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0d1117; }
[data-testid="stSidebar"]          { background-color: #161b22; }
body, p, span, div { color: #e6edf3; }
.section-title {
    font-size:15px; font-weight:600; color:#58a6ff;
    border-left:3px solid #a371f7; padding-left:10px; margin:20px 0 12px;
}
.insight-box {
    background:#161b22; border:1px solid #30363d; border-radius:8px;
    padding:14px 18px; margin:8px 0;
}
.insight-box b { color:#58a6ff; }
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
RAW  = os.path.join(BASE, "data", "raw")

@st.cache_data(ttl=600)
def load():
    daily  = pd.read_csv(os.path.join(PROC, "sensor_daily.csv"))
    daily["date"] = pd.to_datetime(daily["date"])
    master = pd.read_csv(os.path.join(PROC, "machine_master.csv"))
    status = pd.read_csv(os.path.join(PROC, "machine_status.csv"))
    energy = pd.read_csv(os.path.join(PROC, "energy_monthly.csv"))
    alarms = pd.read_csv(os.path.join(RAW,  "alarm_history.csv"), parse_dates=["timestamp"])
    return daily, master, status, energy, alarms

daily, master, status, energy, alarms = load()

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Analytics")
    sel_machine = st.selectbox("Mesin (untuk analitik detail)",
                                sorted(master["machine_id"].tolist()))
    date_max = daily["date"].max().date()
    date_min = daily["date"].min().date()
    sel_start = st.date_input("Mulai", date_max - pd.Timedelta(days=89),
                               min_value=date_min, max_value=date_max)
    sel_end   = st.date_input("Akhir", date_max,
                               min_value=date_min, max_value=date_max)
    st.divider()
    st.page_link("Home.py", label="← Kembali ke Home")

d_start = pd.Timestamp(sel_start)
d_end   = pd.Timestamp(sel_end)

# ── Header ───────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1a0f2e,#0d0819);border-radius:12px;
     padding:20px 28px;margin-bottom:18px;border:1px solid #2d1b5e;">
  <h2 style="color:#e6edf3;margin:0;font-size:22px;">📊 Industrial Analytics & Insight Engine</h2>
  <p style="color:#9ca3af;margin:4px 0 0;font-size:13px;">
    Correlation · Heatmap · Anomaly Detection · Failure Prediction · Energy Forecast
  </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# SECTION 1: Correlation Matrix
# ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔗 Correlation Matrix — Semua Parameter Sensor</div>',
            unsafe_allow_html=True)

daily_m = daily[(daily["machine_id"] == sel_machine) &
                (daily["date"] >= d_start) & (daily["date"] <= d_end)]

num_cols = ["avg_voltage","avg_current","avg_temperature","max_temperature",
            "avg_vibration","max_vibration","avg_humidity",
            "avg_pressure","avg_pf","avg_power_kw","total_energy_kwh"]
nice_labels = ["Voltage","Current","Avg Temp","Max Temp",
               "Avg Vibration","Max Vibration","Humidity",
               "Pressure","Power Factor","Power kW","Energy kWh"]

if len(daily_m) > 10:
    corr = daily_m[num_cols].corr().round(2)
    corr.columns = nice_labels
    corr.index   = nice_labels
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=nice_labels, y=nice_labels,
        colorscale=[[0,"#f85149"],[0.5,"#0d1117"],[1,"#1f6feb"]],
        zmid=0, zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont_size=11,
        colorbar=dict(title="r", tickfont=dict(color="#e6edf3")),
    ))
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=460, margin=dict(l=0,r=0,t=10,b=0),
        font_color="#e6edf3",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Insight otomatis
    corr_series = corr.abs().unstack()
    corr_series = corr_series[corr_series < 1.0].sort_values(ascending=False)
    top_corr    = corr_series.head(3)
    st.markdown(f"""
    <div class="insight-box">
    💡 <b>Auto Insight [{sel_machine}]:</b><br>
    Korelasi terkuat pada mesin ini: {' | '.join(
        [f"<b>{a}</b> ↔ <b>{b}</b> (r={v:.2f})" for (a,b),v in top_corr.items()]
    )}
    </div>""", unsafe_allow_html=True)
else:
    st.info("Data tidak cukup untuk membuat correlation matrix. Perluas rentang tanggal.")

# ──────────────────────────────────────────────────────────
# SECTION 2: Scatter — Hubungan Usia Mesin vs Vibrasi (ATURAN 2)
# ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📐 Validasi Aturan Fisika: Usia Mesin vs Vibrasi Rata-rata</div>',
            unsafe_allow_html=True)

age_vib = status.merge(master[["machine_id","install_year","rated_power_kw"]], on="machine_id")
age_vib["age"] = 2024 - age_vib["install_year"]

cola, colb = st.columns(2)
with cola:
    fig_sc = px.scatter(
        age_vib, x="age", y="avg_vib_30d",
        color="machine_type", size="rated_power_kw",
        hover_data=["machine_id","health_score"],
        trendline="ols",
        labels={"age":"Usia Mesin (Tahun)","avg_vib_30d":"Avg Vibrasi (mm/s)",
                "machine_type":"Tipe","rated_power_kw":"Daya (kW)"},
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_sc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=320, margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", y=1.05, font_size=10),
    )
    fig_sc.update_xaxes(gridcolor="#21262d", ticksuffix=" th")
    fig_sc.update_yaxes(gridcolor="#21262d", ticksuffix=" mm/s")
    st.plotly_chart(fig_sc, use_container_width=True)

    # Korelasi pearson
    r = np.corrcoef(age_vib["age"], age_vib["avg_vib_30d"])[0,1]
    st.markdown(f"""
    <div class="insight-box">
    ✅ <b>Rule Validation:</b> Korelasi Pearson antara usia mesin dan rata-rata vibrasi = <b>{r:.3f}</b><br>
    {"→ Korelasi positif kuat: mesin lebih tua = vibrasi lebih tinggi ✅" if r > 0.4 else
     "→ Korelasi moderat. Faktor lain (tipe mesin, kapasitas) juga berpengaruh."}
    </div>""", unsafe_allow_html=True)

with colb:
    st.markdown("**Health Score vs Jumlah Alarm (30 hari)**")
    fig_hs = px.scatter(
        status, x="alarm_count_30d", y="health_score",
        color="machine_type", hover_data=["machine_id"],
        labels={"alarm_count_30d":"Jumlah Alarm (30 hari)","health_score":"Health Score (%)"},
        trendline="ols", template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_hs.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=320, margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", y=1.05, font_size=10),
    )
    fig_hs.update_xaxes(gridcolor="#21262d")
    fig_hs.update_yaxes(gridcolor="#21262d", ticksuffix="%")
    st.plotly_chart(fig_hs, use_container_width=True)

# ──────────────────────────────────────────────────────────
# SECTION 3: Anomaly Detection — Isolation Forest
# ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🤖 Anomaly Detection — Isolation Forest</div>',
            unsafe_allow_html=True)

feat_cols = ["avg_temperature","avg_vibration","avg_current","avg_pf"]
df_feat = daily_m[feat_cols].dropna()

if len(df_feat) >= 30:
    scaler = StandardScaler()
    X = scaler.fit_transform(df_feat)
    iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    preds = iso.fit_predict(X)

    df_anom = daily_m.loc[df_feat.index].copy()
    df_anom["anomaly"]       = preds
    df_anom["anomaly_score"] = iso.score_samples(X)
    df_anom["label"]         = df_anom["anomaly"].map({1:"Normal", -1:"Anomaly"})

    n_anomaly = (df_anom["anomaly"] == -1).sum()

    fig_anom = go.Figure()
    for label, color, size in [("Normal","#1f6feb",5), ("Anomaly","#f85149",10)]:
        sub = df_anom[df_anom["label"] == label]
        fig_anom.add_trace(go.Scatter(
            x=sub["date"], y=sub["avg_temperature"],
            mode="markers", name=label,
            marker=dict(color=color, size=size,
                        symbol="circle" if label=="Normal" else "x"),
        ))
    fig_anom.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark", height=300, margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", y=1.05),
        xaxis_title="Tanggal", yaxis_title="Suhu (°C)",
    )
    fig_anom.update_yaxes(gridcolor="#21262d")
    fig_anom.update_xaxes(gridcolor="#21262d")
    st.plotly_chart(fig_anom, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
    🤖 <b>Isolation Forest hasil [{sel_machine}]:</b><br>
    Terdeteksi <b style="color:#f85149">{n_anomaly} titik anomali</b> dari {len(df_anom)} data
    ({n_anomaly/len(df_anom)*100:.1f}%) pada periode {d_start.date()} – {d_end.date()}.<br>
    Fitur: Suhu, Vibrasi, Arus, Power Factor (Contamination=5%)
    </div>""", unsafe_allow_html=True)
else:
    st.info("Butuh ≥30 hari data untuk anomaly detection. Perluas rentang tanggal.")

# ──────────────────────────────────────────────────────────
# SECTION 4: Energy Forecast (Linear Regression)
# ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔮 Forecast Konsumsi Energi (Linear Regression)</div>',
            unsafe_allow_html=True)

energy_all = energy.groupby("month")["energy_kwh"].sum().reset_index()
energy_all["month_idx"] = range(len(energy_all))

if len(energy_all) >= 6:
    X_e = energy_all["month_idx"].values.reshape(-1,1)
    y_e = energy_all["energy_kwh"].values
    lr  = LinearRegression().fit(X_e, y_e)
    r2  = lr.score(X_e, y_e)

    # Forecast 3 bulan ke depan
    future_idx = np.array(range(len(energy_all), len(energy_all)+3)).reshape(-1,1)
    forecast   = lr.predict(future_idx)
    from dateutil.relativedelta import relativedelta
    last_month = pd.Period(energy_all["month"].iloc[-1], freq="M")
    future_months = [(last_month + i + 1).strftime("%Y-%m") for i in range(3)]

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=energy_all["month"], y=energy_all["energy_kwh"],
        name="Aktual", line=dict(color="#1f6feb",width=2.5),
        mode="lines+markers", marker_size=6,
    ))
    trend_y = lr.predict(X_e)
    fig_fc.add_trace(go.Scatter(
        x=energy_all["month"], y=trend_y,
        name="Tren", line=dict(color="#d29922",width=2,dash="dash"),
    ))
    fig_fc.add_trace(go.Scatter(
        x=future_months, y=forecast,
        name="Forecast", line=dict(color="#f85149",width=2.5,dash="dot"),
        mode="lines+markers", marker=dict(size=9, symbol="diamond",color="#f85149"),
    ))
    fig_fc.add_vrect(
        x0=future_months[0], x1=future_months[-1],
        fillcolor="rgba(248,81,73,0.05)", line_width=0,
    )
    fig_fc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark", height=300, margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", y=1.05),
        yaxis_title="Total kWh", xaxis_title="Bulan",
    )
    fig_fc.update_yaxes(gridcolor="#21262d", ticksuffix=" kWh")
    fig_fc.update_xaxes(gridcolor="#21262d", tickangle=-30)
    st.plotly_chart(fig_fc, use_container_width=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    for col, month, val in zip([col_f1,col_f2,col_f3], future_months, forecast):
        col.markdown(f"""
        <div class="kpi" style="background:#161b22;border:1px solid #30363d;
             border-radius:8px;padding:14px;text-align:center;">
          <div style="font-size:22px;font-weight:700;color:#f85149">{val:,.0f} kWh</div>
          <div style="font-size:11px;color:#8b949e">Forecast {month}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="insight-box" style="margin-top:10px">
    📈 <b>Model:</b> Linear Regression  |  R² = <b>{r2:.3f}</b><br>
    Slope: <b>{lr.coef_[0]:,.0f} kWh/bulan</b> — Tren konsumsi energi
    {"meningkat ↑" if lr.coef_[0] > 0 else "menurun ↓"} setiap bulan.
    </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# SECTION 5: Failure Risk Score — Random Forest
# ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">⚠️ Failure Risk Score — Machine Learning Ranking</div>',
            unsafe_allow_html=True)

# Feature engineering dari machine_status
risk_df = status.merge(master[["machine_id","install_year","rated_power_kw"]], on="machine_id")
risk_df["age"] = 2024 - risk_df["install_year"]

features = ["age","avg_temp_30d","avg_vib_30d","avg_pf_30d","alarm_count_30d","rated_power_kw"]
# Label: 1 = berisiko tinggi (health score < 60), 0 = aman
risk_df["risk_label"] = (risk_df["health_score"] < 60).astype(int)

X_risk = risk_df[features].fillna(0)
y_risk = risk_df["risk_label"]

if y_risk.sum() >= 2:
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_risk, y_risk)
    risk_df["failure_prob"] = rf.predict_proba(X_risk)[:,1]
else:
    # Fallback: gunakan inverse health score
    risk_df["failure_prob"] = (100 - risk_df["health_score"]) / 100

risk_df = risk_df.sort_values("failure_prob", ascending=False)
colors_risk = risk_df["failure_prob"].apply(
    lambda x: "#f85149" if x > 0.6 else ("#d29922" if x > 0.35 else "#2ea043")
).tolist()

fig_risk = go.Figure(go.Bar(
    x=risk_df["machine_id"],
    y=(risk_df["failure_prob"] * 100).round(1),
    marker_color=colors_risk,
    text=(risk_df["failure_prob"] * 100).round(0).astype(int).astype(str) + "%",
    textposition="outside",
))
fig_risk.add_hline(y=60, line_dash="dash", line_color="#f85149",
                    annotation_text="Risiko Tinggi >60%")
fig_risk.add_hline(y=35, line_dash="dash", line_color="#d29922",
                    annotation_text="Risiko Sedang >35%")
fig_risk.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    template="plotly_dark", height=330, margin=dict(l=0,r=0,t=10,b=60),
    yaxis_title="Failure Probability (%)", yaxis_range=[0,115],
)
fig_risk.update_xaxes(gridcolor="#21262d", tickangle=-40)
fig_risk.update_yaxes(gridcolor="#21262d", ticksuffix="%")
st.plotly_chart(fig_risk, use_container_width=True)

# Feature importance
if y_risk.sum() >= 2:
    importance = pd.DataFrame({
        "Fitur": features,
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig_imp = go.Figure(go.Bar(
        x=importance["Importance"], y=importance["Fitur"],
        orientation="h", marker_color="#a371f7",
        text=importance["Importance"].apply(lambda x: f"{x:.3f}"),
        textposition="outside",
    ))
    fig_imp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_dark", height=250, margin=dict(l=0,r=0,t=10,b=0),
        xaxis_title="Feature Importance",
    )
    fig_imp.update_xaxes(gridcolor="#21262d")
    fig_imp.update_yaxes(gridcolor="#21262d")
    st.markdown("**Feature Importance — Random Forest**")
    st.plotly_chart(fig_imp, use_container_width=True)

# ── Tabel Prioritas Maintenance ──────────────────────────
st.markdown('<div class="section-title">🎯 Rekomendasi Prioritas Maintenance</div>',
            unsafe_allow_html=True)
priority = risk_df[["machine_id","machine_type","health_score","avg_temp_30d",
                      "avg_vib_30d","alarm_count_30d","failure_prob"]].copy()
priority["failure_prob_%"] = (priority["failure_prob"] * 100).round(1)
priority["Prioritas"] = priority["failure_prob"].apply(
    lambda x: "🔴 CRITICAL — Segera" if x > 0.6 else
              ("🟡 HIGH — Minggu Ini"  if x > 0.35 else
               ("🟠 MEDIUM — Bulan Ini" if x > 0.15 else "🟢 LOW — Terjadwal")))
priority = priority.sort_values("failure_prob", ascending=False).reset_index(drop=True)
priority.index += 1
st.dataframe(
    priority[["machine_id","machine_type","health_score","avg_temp_30d",
              "avg_vib_30d","alarm_count_30d","failure_prob_%","Prioritas"]].rename(columns={
        "machine_id":"Mesin","machine_type":"Tipe","health_score":"Health %",
        "avg_temp_30d":"Avg Temp","avg_vib_30d":"Avg Vib",
        "alarm_count_30d":"Alarm 30hr","failure_prob_%":"Risk %",
    }),
    use_container_width=True, hide_index=False, height=480,
)

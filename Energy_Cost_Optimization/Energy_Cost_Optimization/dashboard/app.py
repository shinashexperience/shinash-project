"""
Energy Cost Optimization Dashboard
PT Surya Textile Indonesia (Simulasi)
Tool: Streamlit

Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Optimization | PT Surya Textile",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Brand Colours ─────────────────────────────────────────────────────────────
NAVY   = "#1B3A6B"
ORANGE = "#E87722"
GREEN  = "#2E7D32"
RED    = "#C62828"
GRAY   = "#607D8B"

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #1B3A6B; }
    [data-testid="stSidebar"] * { color: white !important; }
    .metric-card {
        background: linear-gradient(135deg, #1B3A6B 0%, #2c5aa0 100%);
        padding: 20px; border-radius: 12px; text-align: center; color: white;
        box-shadow: 0 4px 12px rgba(27,58,107,0.3);
    }
    .metric-card .label { font-size: 12px; opacity: 0.8; margin-bottom: 6px; }
    .metric-card .value { font-size: 24px; font-weight: 700; }
    .metric-card .unit  { font-size: 11px; opacity: 0.7; margin-top: 3px; }
    .saving-card {
        background: linear-gradient(135deg, #2E7D32 0%, #43a047 100%);
        padding: 16px; border-radius: 12px; text-align: center; color: white;
        box-shadow: 0 4px 12px rgba(46,125,50,0.3);
    }
    .warn-card {
        background: linear-gradient(135deg, #C62828 0%, #e53935 100%);
        padding: 16px; border-radius: 12px; text-align: center; color: white;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 14px; font-weight: 600; padding: 10px 20px;
    }
    h1 { color: #1B3A6B !important; }
    h2 { color: #1B3A6B !important; }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(base, "..")
    clean = os.path.join(root, "data", "processed", "clean_energy_data.csv")
    df = pd.read_csv(clean, parse_dates=["timestamp"])

    machine_summary = (df.groupby(["machine_id", "machine_type"])
                         .agg(total_energy_kwh=("energy_kwh","sum"),
                              total_cost_idr=("cost_idr","sum"),
                              avg_pf=("power_factor","mean"),
                              avg_power_kw=("active_power_kw","mean"))
                         .reset_index()
                         .sort_values("total_energy_kwh", ascending=False))

    hourly = (df.groupby("hour")
                .agg(avg_energy=("energy_kwh","mean"),
                     avg_cost=("cost_idr","mean"),
                     total_cost=("cost_idr","sum"))
                .reset_index())

    monthly = (df.groupby("month")
                 .agg(total_energy=("energy_kwh","sum"),
                      total_cost=("cost_idr","sum"))
                 .reset_index())
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["month_name"] = [month_names[m-1] for m in monthly["month"]]

    daily = (df.groupby("day_of_week")
               .agg(avg_cost=("cost_idr","mean"),
                    total_cost=("cost_idr","sum"))
               .reset_index())

    with open(os.path.join(root, "reports", "simulation.json")) as f:
        sim = json.load(f)

    return df, machine_summary, hourly, monthly, daily, sim

df, ms, hourly, monthly, daily, sim = load_data()

total_energy  = df["energy_kwh"].sum()
total_cost    = df["cost_idr"].sum()
avg_daily     = df.groupby("date")["cost_idr"].sum().mean()
bad_pf_pct    = (df["power_factor"] < 0.80).mean() * 100

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://via.placeholder.com/200x60/1B3A6B/ffffff?text=PT+Surya+Textile",
                 width=200)
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Energy Dashboard")
st.sidebar.markdown("**PT Surya Textile Indonesia**")
st.sidebar.markdown("Periode: Jan – Des 2025")
st.sidebar.markdown("---")

machine_filter = st.sidebar.multiselect(
    "Filter Tipe Mesin:",
    options=["Motor", "Pump", "Compressor"],
    default=["Motor", "Pump", "Compressor"],
)
month_filter = st.sidebar.slider("Filter Bulan:", 1, 12, (1, 12))

df_filtered = df[
    (df["machine_type"].isin(machine_filter)) &
    (df["month"] >= month_filter[0]) &
    (df["month"] <= month_filter[1])
]
ms_filtered = ms[ms["machine_type"].isin(machine_filter)]

st.sidebar.markdown("---")
st.sidebar.markdown(f"📊 **Records:** {len(df_filtered):,}")
st.sidebar.markdown(f"🔧 **Mesin:** {df_filtered['machine_id'].nunique()}")
st.sidebar.markdown("---")
st.sidebar.markdown("💡 *Dashboard v1.0*")

# ─── Main Header ──────────────────────────────────────────────────────────────
st.markdown(
    f"<h1 style='color:{NAVY};margin-bottom:4px;'>⚡ Energy Cost Optimization Dashboard</h1>"
    f"<p style='color:{GRAY};font-size:14px;margin-top:0;'>"
    f"PT Surya Textile Indonesia (Simulasi) &nbsp;|&nbsp; Periode: Januari – Desember 2025 "
    f"&nbsp;|&nbsp; 30 Mesin Produksi</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Summary",
    "🔧 Machine Analysis",
    "⚡ Power Quality",
    "💰 Savings Simulation",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("KPI Utama Tahun 2025")

    # KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>TOTAL ENERGI 2025</div>
            <div class='value'>{total_energy/1e6:.2f} GWh</div>
            <div class='unit'>{total_energy:,.0f} kWh</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>TOTAL BIAYA 2025</div>
            <div class='value'>Rp {total_cost/1e9:.2f} M</div>
            <div class='unit'>Rp {total_cost:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>RATA-RATA BIAYA/HARI</div>
            <div class='value'>Rp {avg_daily/1e6:.1f} Jt</div>
            <div class='unit'>Rp {avg_daily:,.0f}/hari</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        color = 'warn-card' if bad_pf_pct > 5 else 'saving-card'
        st.markdown(f"""<div class='{color}'>
            <div class='label'>INTERVAL PF BURUK</div>
            <div class='value'>{bad_pf_pct:.1f}%</div>
            <div class='unit'>PF &lt; 0,80 (target &lt; 3%)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly trend
    col_a, col_b = st.columns(2)
    with col_a:
        fig_monthly = make_subplots(specs=[[{"secondary_y": True}]])
        fig_monthly.add_trace(
            go.Bar(x=monthly["month_name"], y=monthly["total_energy"]/1000,
                   name="Energi (MWh)", marker_color=NAVY, opacity=0.8),
            secondary_y=False,
        )
        fig_monthly.add_trace(
            go.Scatter(x=monthly["month_name"], y=monthly["total_cost"]/1e9,
                       name="Biaya (Rp Milyar)", line=dict(color=ORANGE, width=2.5),
                       mode="lines+markers", marker=dict(size=7)),
            secondary_y=True,
        )
        fig_monthly.update_layout(
            title="Tren Energi & Biaya Bulanan",
            plot_bgcolor="#F8F9FA", paper_bgcolor="white",
            font=dict(size=11), height=360,
            legend=dict(orientation="h", y=1.1),
        )
        fig_monthly.update_yaxes(title_text="Energi (MWh)", secondary_y=False)
        fig_monthly.update_yaxes(title_text="Biaya (Rp Milyar)", secondary_y=True)
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col_b:
        # Peak vs Normal pie + Type bar
        peak_c  = df[df["is_peak"]==1]["cost_idr"].sum()
        norm_c  = df[df["is_peak"]==0]["cost_idr"].sum()
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Peak (17–22 hrs)", "Normal Hours"],
            values=[peak_c, norm_c],
            hole=0.42,
            marker=dict(colors=[RED, NAVY], line=dict(color="white", width=2)),
        )])
        fig_pie.update_layout(
            title="Proporsi Biaya: Peak vs Normal Hour",
            plot_bgcolor="#F8F9FA", paper_bgcolor="white",
            height=360, font=dict(size=11),
            annotations=[dict(text=f"Total<br>Rp {total_cost/1e9:.2f}M",
                              x=0.5, y=0.5, font_size=12, showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Hourly heatmap-style bar
    fig_hr = go.Figure()
    colors_hr = [RED if 17 <= h < 22 else NAVY for h in hourly["hour"]]
    fig_hr.add_trace(go.Bar(
        x=hourly["hour"], y=hourly["avg_cost"],
        marker_color=colors_hr, name="Avg Cost/Interval",
        text=[f"Rp {v:,.0f}" for v in hourly["avg_cost"]],
        textposition="outside", textfont=dict(size=8),
    ))
    fig_hr.add_vrect(x0=16.5, x1=21.5, fillcolor=RED, opacity=0.08,
                     annotation_text="Peak Hours", annotation_position="top left")
    fig_hr.update_layout(
        title="Rata-rata Biaya per Jam (Merah = Peak Hour Tarif Rp 1.700/kWh)",
        xaxis_title="Jam", yaxis_title="Avg Cost (IDR)",
        plot_bgcolor="#F8F9FA", paper_bgcolor="white",
        height=320, font=dict(size=10),
        xaxis=dict(tickvals=list(range(0,24)),
                   ticktext=[f"{h:02d}:00" for h in range(24)]),
    )
    st.plotly_chart(fig_hr, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – MACHINE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Analisis Konsumsi & Biaya per Mesin")

    top_n = st.slider("Tampilkan Top N mesin:", 5, 30, 10, key="topn")

    col1, col2 = st.columns(2)

    # Top consumers
    with col1:
        top_e = ms_filtered.nlargest(top_n, "total_energy_kwh")
        type_colors = {"Motor": NAVY, "Pump": ORANGE, "Compressor": GREEN}
        colors_e = [type_colors.get(t, GRAY) for t in top_e["machine_type"]]

        fig_te = go.Figure(go.Bar(
            y=top_e["machine_id"], x=top_e["total_energy_kwh"],
            orientation="h", marker_color=colors_e,
            text=[f"{v:,.0f} kWh" for v in top_e["total_energy_kwh"]],
            textposition="outside", textfont=dict(size=9),
        ))
        fig_te.update_layout(
            title=f"Top {top_n} – Konsumsi Energi (kWh)",
            xaxis_title="Total kWh", height=400,
            plot_bgcolor="#F8F9FA", paper_bgcolor="white",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_te, use_container_width=True)

    # Top cost
    with col2:
        top_c = ms_filtered.nlargest(top_n, "total_cost_idr")
        colors_c = [type_colors.get(t, GRAY) for t in top_c["machine_type"]]

        fig_tc = go.Figure(go.Bar(
            y=top_c["machine_id"], x=top_c["total_cost_idr"]/1e6,
            orientation="h", marker_color=colors_c,
            text=[f"Rp {v/1e6:.1f}M" for v in top_c["total_cost_idr"]],
            textposition="outside", textfont=dict(size=9),
        ))
        fig_tc.update_layout(
            title=f"Top {top_n} – Total Biaya Listrik (IDR Juta)",
            xaxis_title="IDR Juta", height=400,
            plot_bgcolor="#F8F9FA", paper_bgcolor="white",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_tc, use_container_width=True)

    # Machine type summary
    type_sum = (ms_filtered.groupby("machine_type")
                            .agg(total_energy=("total_energy_kwh","sum"),
                                 total_cost=("total_cost_idr","sum"),
                                 count=("machine_id","count"))
                            .reset_index())

    fig_type = px.bar(
        type_sum, x="machine_type",
        y=["total_energy", "total_cost"],
        barmode="group",
        color_discrete_map={"total_energy": NAVY, "total_cost": ORANGE},
        title="Perbandingan Total Energi vs. Biaya per Tipe Mesin",
        labels={"value": "Nilai", "machine_type": "Tipe Mesin", "variable": "Metrik"},
    )
    fig_type.update_layout(plot_bgcolor="#F8F9FA", paper_bgcolor="white", height=300)
    st.plotly_chart(fig_type, use_container_width=True)

    # Data table
    st.subheader("Tabel Detail per Mesin")
    display_ms = ms_filtered.copy()
    display_ms["total_energy_kwh"] = display_ms["total_energy_kwh"].round(0)
    display_ms["total_cost_idr"]   = display_ms["total_cost_idr"].apply(lambda x: f"Rp {x:,.0f}")
    display_ms["avg_pf"]           = display_ms["avg_pf"].round(4)
    display_ms["avg_power_kw"]     = display_ms["avg_power_kw"].round(3)
    display_ms.columns = ["Machine ID","Type","Total Energy (kWh)",
                          "Total Cost (IDR)","Avg PF","Avg Power (kW)"]
    st.dataframe(display_ms, use_container_width=True, height=400)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – POWER QUALITY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Analisis Kualitas Daya (Power Factor)")

    c1, c2 = st.columns([3, 2])
    with c1:
        pf_rank = ms_filtered.sort_values("avg_pf")
        pf_colors = [RED if pf < 0.80 else (ORANGE if pf < 0.85 else GREEN)
                     for pf in pf_rank["avg_pf"]]

        fig_pf = go.Figure(go.Bar(
            y=pf_rank["machine_id"], x=pf_rank["avg_pf"],
            orientation="h", marker_color=pf_colors,
            text=[f"{v:.4f}" for v in pf_rank["avg_pf"]],
            textposition="outside", textfont=dict(size=8.5),
        ))
        fig_pf.add_vline(x=0.85, line_dash="dash", line_color=ORANGE,
                         annotation_text="Target 0.85", annotation_position="top right")
        fig_pf.add_vline(x=0.80, line_dash="dot", line_color=RED,
                         annotation_text="Batas Kritis 0.80", annotation_position="bottom right")
        fig_pf.update_layout(
            title="Power Factor Ranking – Semua Mesin",
            xaxis_title="Average Power Factor", xaxis=dict(range=[0.55, 1.0]),
            height=max(350, len(pf_rank)*22),
            plot_bgcolor="#F8F9FA", paper_bgcolor="white",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_pf, use_container_width=True)

    with c2:
        # PF category distribution
        pf_sample = df_filtered["power_factor"].sample(min(50000, len(df_filtered)),
                                                        random_state=42)
        bins = [0.5, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
        labels = ["< 0.75\n(Kritis)", "0.75–0.80\n(Buruk)", "0.80–0.85\n(Perlu Perhatian)",
                  "0.85–0.90\n(Baik)", "0.90–0.95\n(Sangat Baik)", "> 0.95\n(Optimal)"]
        pf_binned = pd.cut(pf_sample, bins=bins, labels=labels).value_counts()

        fig_pf_dist = go.Figure(go.Bar(
            x=labels, y=pf_binned.values,
            marker_color=[RED, RED, ORANGE, GREEN, GREEN, GREEN],
            text=pf_binned.values, textposition="outside",
        ))
        fig_pf_dist.update_layout(
            title="Distribusi Power Factor",
            xaxis_title="Kategori PF", yaxis_title="Jumlah Interval",
            height=350, plot_bgcolor="#F8F9FA", paper_bgcolor="white",
            font=dict(size=9),
        )
        st.plotly_chart(fig_pf_dist, use_container_width=True)

        # PF stats
        st.metric("PF Rata-rata Seluruh Mesin",
                  f"{df_filtered['power_factor'].mean():.4f}")
        st.metric("% Interval PF Buruk (< 0.80)",
                  f"{(df_filtered['power_factor']<0.80).mean()*100:.2f}%",
                  delta="-6.7% target", delta_color="inverse")
        st.metric("Mesin Paling Bermasalah",
                  pf_rank.iloc[0]["machine_id"],
                  f"PF = {pf_rank.iloc[0]['avg_pf']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – SAVINGS SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Simulasi Skenario Penghematan Energi")

    base  = sim["baseline"]
    sa    = sim["scenario_a"]
    sb    = sim["scenario_b"]
    sc    = sim["scenario_c"]
    comb  = sim["combined"]

    # Current cost KPI
    st.markdown(f"""
    <div class='metric-card' style='margin-bottom:16px;'>
        <div class='label'>BIAYA LISTRIK TAHUNAN (BASELINE)</div>
        <div class='value'>Rp {base['total_cost_idr']/1e9:.3f} Milyar</div>
        <div class='unit'>Total: Rp {base['total_cost_idr']:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    # Scenario cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='saving-card'>
            <div class='label'>SCENARIO A – PF Correction</div>
            <div class='value'>Rp {sa['saving_idr']/1e6:.1f} Jt</div>
            <div class='unit'>Saving: {sa['saving_pct']:.2f}%<br>
            Biaya Baru: Rp {sa['new_annual_cost']/1e9:.3f}M</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='saving-card'>
            <div class='label'>SCENARIO B – Load Shift 18→13</div>
            <div class='value'>Rp {sb['saving_idr']/1e6:.1f} Jt</div>
            <div class='unit'>Saving: {sb['saving_pct']:.2f}%<br>
            Biaya Baru: Rp {sb['new_annual_cost']/1e9:.3f}M</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='saving-card'>
            <div class='label'>SCENARIO C – 5% Reduction</div>
            <div class='value'>Rp {sc['saving_idr']/1e6:.1f} Jt</div>
            <div class='unit'>Saving: {sc['saving_pct']:.2f}%<br>
            Biaya Baru: Rp {sc['new_annual_cost']/1e9:.3f}M</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='saving-card' style='background:linear-gradient(135deg,#1B5E20,#2E7D32)'>
            <div class='label'>TOTAL GABUNGAN (A+B+C)</div>
            <div class='value'>Rp {comb['saving_idr']/1e6:.1f} Jt</div>
            <div class='unit'>Saving: {comb['saving_pct']:.2f}%<br>
            Biaya Baru: Rp {comb['new_annual_cost']/1e9:.3f}M</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Waterfall chart
    fig_wf = go.Figure(go.Waterfall(
        name="Cost Flow",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Baseline", "Scenario A\n(PF Correction)", "Scenario B\n(Load Shift)",
           "Scenario C\n(5% Reduction)", "Total Setelah\nSemua Skenario"],
        textposition="outside",
        text=[f"Rp {base['total_cost_idr']/1e9:.3f}M",
              f"-Rp {sa['saving_idr']/1e6:.1f}Jt",
              f"-Rp {sb['saving_idr']/1e6:.1f}Jt",
              f"-Rp {sc['saving_idr']/1e6:.1f}Jt",
              f"Rp {comb['new_annual_cost']/1e9:.3f}M"],
        y=[base["total_cost_idr"]/1e9,
           -sa["saving_idr"]/1e9,
           -sb["saving_idr"]/1e9,
           -sc["saving_idr"]/1e9,
           comb["new_annual_cost"]/1e9],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": RED}},
        decreasing={"marker": {"color": GREEN}},
        totals={"marker": {"color": NAVY}},
    ))
    fig_wf.update_layout(
        title="Waterfall Chart – Dampak Penghematan per Skenario (Rp Milyar)",
        yaxis_title="Biaya (Rp Milyar)",
        plot_bgcolor="#F8F9FA", paper_bgcolor="white",
        height=420, font=dict(size=11),
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # Bar comparison
    scenarios_df = pd.DataFrame({
        "Skenario": ["Baseline", "A: PF\nCorrection", "B: Load\nShift 18→13",
                     "C: 5%\nReduction", "Gabungan\nA+B+C"],
        "Biaya (Rp M)": [
            base["total_cost_idr"]/1e9,
            sa["new_annual_cost"]/1e9,
            sb["new_annual_cost"]/1e9,
            sc["new_annual_cost"]/1e9,
            comb["new_annual_cost"]/1e9,
        ],
        "Penghematan (Rp M)": [
            0,
            sa["saving_idr"]/1e9,
            sb["saving_idr"]/1e9,
            sc["saving_idr"]/1e9,
            comb["saving_idr"]/1e9,
        ]
    })
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="Biaya Setelah Skenario",
        x=scenarios_df["Skenario"], y=scenarios_df["Biaya (Rp M)"],
        marker_color=[GRAY, GREEN, ORANGE, NAVY, RED],
        text=[f"Rp {v:.3f}M" for v in scenarios_df["Biaya (Rp M)"]],
        textposition="outside",
    ))
    fig_bar.update_layout(
        title="Perbandingan Biaya Tahunan per Skenario",
        yaxis_title="Biaya Tahunan (Rp Milyar)",
        plot_bgcolor="#F8F9FA", paper_bgcolor="white",
        height=360, font=dict(size=10),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Summary table
    st.subheader("Ringkasan Finansial Semua Skenario")
    summary_tbl = pd.DataFrame({
        "Skenario": ["A – PF Correction", "B – Load Shift 18→13",
                     "C – 5% Energy Reduction", "KOMBINASI A+B+C"],
        "Penghematan/Tahun": [
            f"Rp {sa['saving_idr']:,.0f}",
            f"Rp {sb['saving_idr']:,.0f}",
            f"Rp {sc['saving_idr']:,.0f}",
            f"Rp {comb['saving_idr']:,.0f}",
        ],
        "% Penghematan": [
            f"{sa['saving_pct']:.2f}%",
            f"{sb['saving_pct']:.2f}%",
            f"{sc['saving_pct']:.2f}%",
            f"{comb['saving_pct']:.2f}%",
        ],
        "Biaya Setelah Skenario": [
            f"Rp {sa['new_annual_cost']:,.0f}",
            f"Rp {sb['new_annual_cost']:,.0f}",
            f"Rp {sc['new_annual_cost']:,.0f}",
            f"Rp {comb['new_annual_cost']:,.0f}",
        ],
    })
    st.dataframe(summary_tbl, use_container_width=True, hide_index=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<p style='color:{GRAY};font-size:11px;text-align:center;'>"
    f"Energy Cost Optimization Dashboard | PT Surya Textile Indonesia (Simulasi) | "
    f"Data Periode: 1 Jan 2025 – 31 Des 2025 | 30 Mesin Produksi | "
    f"Dashboard v1.0</p>",
    unsafe_allow_html=True,
)

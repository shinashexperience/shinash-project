import streamlit as st
import pandas as pd
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def show():
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    with open(f'{BASE}/data/stats.json') as f:
        stats = json.load(f)

    st.title("🏠 Overview — Industrial Monitoring Dashboard")
    st.caption("PT Nusantara Manufacturing | Data: Januari — Desember 2025")
    st.divider()

    # ─── KPIs ────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚡ Total Energy", f"{stats['total_energy']/1000:.1f} GWh",
              delta="Konsumsi 2025")
    c2.metric("🔧 Total Machines", len(stats['machines']),
              delta="Aktif")
    c3.metric("⚠️ WARNING Events", stats['warning_count'],
              delta="1.5% dari total", delta_color="inverse")
    c4.metric("🚨 Anomali", stats['anomaly_count'],
              delta="Isolation Forest", delta_color="inverse")
    st.divider()

    # ─── Health per machine ───────────────────────────────────────────────────
    st.subheader("❤️ Machine Health Summary")
    cols = st.columns(len(stats['machines']))
    cat_color = {'HEALTHY':'🟢', 'WARNING':'🟡', 'CRITICAL':'🔴'}
    for col, m in zip(cols, stats['machines']):
        sc  = stats['health_scores'][m]
        cat = stats['health_cat'][m]
        icon = cat_color.get(cat, '⚪')
        col.metric(f"{icon} {m}", f"{sc}%", delta=cat,
                   delta_color="normal" if cat=='HEALTHY' else "inverse")
    st.divider()

    # ─── Warning distribution ─────────────────────────────────────────────────
    st.subheader("⚠️ Warning Events per Machine")
    w_df = pd.DataFrame(list(stats['warning_by_machine'].items()),
                        columns=['Machine','Warning Events'])
    st.bar_chart(w_df.set_index('Machine'))

    # ─── Summary table ────────────────────────────────────────────────────────
    st.subheader("📋 Summary Table")
    summary = []
    for m in stats['machines']:
        summary.append({
            'Machine':       m,
            'Health Score':  f"{stats['health_scores'][m]}%",
            'Category':      stats['health_cat'][m],
            'Failure Risk':  f"{stats['fail_prob'][m]}%",
            'RUL (Days)':    stats['rul'][m],
            'Warnings':      stats['warning_by_machine'].get(m, 0),
            'Energy (kWh)':  f"{stats['energy_by_machine'][m]:,.0f}",
        })
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

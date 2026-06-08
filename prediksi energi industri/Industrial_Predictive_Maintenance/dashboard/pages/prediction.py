import streamlit as st
import pandas as pd
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def show():
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    with open(f'{BASE}/data/stats.json') as f:
        stats = json.load(f)

    st.title("🔮 Predictive Maintenance Dashboard")
    st.caption("Failure Probability | Remaining Useful Life | Anomaly Detection")
    st.divider()

    # ─── Machine cards ────────────────────────────────────────────────────────
    st.subheader("⚡ Machine Status Overview")
    cols = st.columns(len(stats['machines']))
    for col, m in zip(cols, stats['machines']):
        fp  = stats['fail_prob'][m]
        rul = stats['rul'][m]
        sc  = stats['health_scores'][m]
        cat = stats['health_cat'][m]
        risk = 'HIGH' if fp > 15 else 'MEDIUM' if fp > 8 else 'LOW'
        icon = {'HEALTHY':'🟢','WARNING':'🟡','CRITICAL':'🔴'}.get(cat,'⚪')
        with col:
            st.markdown(f"**{icon} {m}**")
            st.metric("Health Score", f"{sc}%")
            st.metric("Failure Risk", f"{fp}%", delta=risk,
                      delta_color="normal" if risk=='LOW' else "inverse")
            st.metric("RUL", f"{rul} Days")
    st.divider()

    # ─── Detailed view ────────────────────────────────────────────────────────
    st.subheader("🔍 Detailed Predictive Analysis")
    sel = st.selectbox("Select Machine for Detail:", stats['machines'])

    fp  = stats['fail_prob'][sel]
    rul = stats['rul'][sel]
    sc  = stats['health_scores'][sel]
    cat = stats['health_cat'][sel]
    risk = 'HIGH' if fp > 15 else 'MEDIUM' if fp > 8 else 'LOW'

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Machine", sel)
    c2.metric("Health Score", f"{sc}%", delta=cat,
              delta_color="normal" if cat=='HEALTHY' else "inverse")
    c3.metric("Failure Risk", risk)
    c4.metric("Remaining Life", f"{rul} Days")
    st.divider()

    # Progress bar for health
    st.subheader(f"❤️ Health Index — {sel}")
    color_map = {'HEALTHY':'green','WARNING':'orange','CRITICAL':'red'}
    st.progress(int(sc), text=f"Health Score: {sc}% ({cat})")

    # Failure probability gauge text
    st.subheader("🎯 Failure Probability")
    st.progress(int(min(fp*5, 100)), text=f"Failure Probability: {fp}% — Risk Level: {risk}")

    # RUL gauge
    st.subheader("⏳ Remaining Useful Life")
    rul_pct = min(int(rul/90*100), 100)
    st.progress(rul_pct, text=f"Remaining Life: {rul} of 90 days max")
    st.divider()

    # ─── Maintenance Schedule ─────────────────────────────────────────────────
    st.subheader("📅 Recommended Maintenance Schedule")
    sched = []
    for m in stats['machines']:
        r = stats['rul'][m]
        c = stats['health_cat'][m]
        fp_v = stats['fail_prob'][m]
        risk_v = 'HIGH' if fp_v > 15 else 'MEDIUM' if fp_v > 8 else 'LOW'
        action = 'Immediate Inspection' if r < 20 else 'Schedule in 45 days' if r < 50 else 'Routine (90 days)'
        sched.append({'Machine':m, 'RUL (Days)':r, 'Category':c,
                      'Risk Level':risk_v, 'Recommended Action':action})
    st.dataframe(pd.DataFrame(sched), use_container_width=True, hide_index=True)

    # ─── Model performance ────────────────────────────────────────────────────
    st.divider()
    st.subheader("🤖 ML Model Performance")
    model_rows = []
    for name, res in stats['model_results'].items():
        model_rows.append({'Model':name,'Accuracy':f"{res['Accuracy']}%",
                           'Precision':f"{res['Precision']}%",
                           'Recall':f"{res['Recall']}%",'F1 Score':f"{res['F1 Score']}%"})
    st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)
    st.caption("Model: Random Forest (n=150) & Logistic Regression | Features: temp, vibration, current, voltage, power_kw")

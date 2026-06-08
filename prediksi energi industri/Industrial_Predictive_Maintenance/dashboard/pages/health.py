import streamlit as st
import pandas as pd
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def show():
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    df = pd.read_csv(f'{BASE}/data/raw_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    with open(f'{BASE}/data/stats.json') as f:
        stats = json.load(f)

    st.title("❤️ Asset Health Monitoring")
    st.caption("Temperature | Vibration | Current — Health Index per Mesin")
    st.divider()

    sel = st.selectbox("Pilih Mesin:", stats['machines'])
    dff = df[df['machine_id']==sel].set_index('timestamp')
    st.divider()

    sc  = stats['health_scores'][sel]
    cat = stats['health_cat'][sel]
    rul = stats['rul'][sel]
    cat_color = {'HEALTHY':'🟢','WARNING':'🟡','CRITICAL':'🔴'}

    c1,c2,c3 = st.columns(3)
    c1.metric(f"{cat_color.get(cat,'⚪')} Health Score", f"{sc}%", delta=cat,
              delta_color="normal" if cat=='HEALTHY' else "inverse")
    c2.metric("🕐 Est. RUL", f"{rul} Hari")
    c3.metric("⚠️ Failure Risk", f"{stats['fail_prob'][sel]}%")
    st.divider()

    # ─── Temp ─────────────────────────────────────────────────────────────────
    st.subheader("🌡️ Temperature Trend")
    temp_w = dff['temperature_c'].resample('D').mean()
    st.line_chart(temp_w)

    # ─── Vibration ────────────────────────────────────────────────────────────
    st.subheader("📳 Vibration Trend")
    vib_w = dff['vibration_mm_s'].resample('D').mean()
    st.line_chart(vib_w)

    # ─── Current ──────────────────────────────────────────────────────────────
    st.subheader("⚡ Current Trend")
    cur_w = dff['current_a'].resample('D').mean()
    st.line_chart(cur_w)

    # ─── Stats table ──────────────────────────────────────────────────────────
    st.subheader(f"📊 Statistical Summary — {sel}")
    stats_df = dff[['temperature_c','vibration_mm_s','current_a','voltage_v','power_kw']].describe().round(2)
    st.dataframe(stats_df, use_container_width=True)

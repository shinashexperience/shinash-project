import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def show():
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    df = pd.read_csv(f'{BASE}/data/raw_data.csv')
    df['timestamp']  = pd.to_datetime(df['timestamp'])
    df['hour']       = df['timestamp'].dt.hour
    df['month']      = df['timestamp'].dt.month
    df['week']       = df['timestamp'].dt.isocalendar().week.astype(int)
    df['day']        = df['timestamp'].dt.date
    df['month_name'] = df['timestamp'].dt.strftime('%b')

    st.title("⚡ Energy Analysis")
    st.caption("Analisis konsumsi energi per mesin, per jam, per minggu, per bulan")
    st.divider()

    sel = st.multiselect("Filter Machine:", df['machine_id'].unique().tolist(),
                         default=df['machine_id'].unique().tolist())
    dff = df[df['machine_id'].isin(sel)] if sel else df
    st.divider()

    # ─── KPIs ────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Energy", f"{dff['power_kw'].sum()/1000:.1f} GWh")
    c2.metric("Avg Power", f"{dff['power_kw'].mean():.1f} kW")
    c3.metric("Peak Hour", f"{dff.groupby('hour')['power_kw'].mean().idxmax()}:00")
    st.divider()

    # ─── Daily ───────────────────────────────────────────────────────────────
    st.subheader("📅 Daily Energy Consumption")
    daily = dff.groupby('day')['power_kw'].sum().reset_index()
    daily.columns = ['Date','Power (kW)']
    st.line_chart(daily.set_index('Date'))

    # ─── Hourly ──────────────────────────────────────────────────────────────
    st.subheader("🕐 Hourly Energy Profile")
    hourly = dff.groupby('hour')['power_kw'].mean().reset_index()
    hourly.columns = ['Hour','Avg Power (kW)']
    st.bar_chart(hourly.set_index('Hour'))

    # ─── Monthly ─────────────────────────────────────────────────────────────
    st.subheader("📆 Monthly Energy Consumption")
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    monthly = dff.groupby('month')['power_kw'].sum().reset_index()
    monthly['month_name'] = monthly['month'].apply(lambda x: month_names[x-1])
    st.bar_chart(monthly.set_index('month_name')['power_kw'])

    # ─── Per Machine ─────────────────────────────────────────────────────────
    st.subheader("🔧 Energy per Machine")
    em = dff.groupby('machine_id')['power_kw'].sum().reset_index()
    em.columns = ['Machine','Total Energy (kWh)']
    st.bar_chart(em.set_index('Machine'))

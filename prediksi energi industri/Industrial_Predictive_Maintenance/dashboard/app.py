"""
Industrial Energy Monitoring & Predictive Maintenance
Streamlit Dashboard — Main Entry Point
"""
import streamlit as st

st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/1565C0/FFFFFF?text=PT+Nusantara", use_column_width=True)
    st.markdown("## ⚙️ Navigation")
    page = st.radio("", [
        "🏠 Overview",
        "⚡ Energy Analysis",
        "❤️ Asset Health",
        "🔮 Prediction",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Industrial Energy Monitoring\n& Predictive Maintenance\nPT Nusantara Manufacturing 2025")

# ─── Route ────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    from pages import overview; overview.show()
elif page == "⚡ Energy Analysis":
    from pages import energy; energy.show()
elif page == "❤️ Asset Health":
    from pages import health; health.show()
elif page == "🔮 Prediction":
    from pages import prediction; prediction.show()

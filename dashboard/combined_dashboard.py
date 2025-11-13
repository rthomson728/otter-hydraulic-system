import streamlit as st

st.set_page_config(page_title="Otter Hydraulic System Dashboard", layout="wide", page_icon="💧")

# --- Smaller title + top spacing fix ---
st.markdown("""
<style>
h1 {
    font-size: 1.4rem !important;      /* slightly smaller title */
    margin-top: 0.5rem !important;     /* adds a little padding from top */
    margin-bottom: 0.7rem !important;
}
[data-testid="stToolbar"] { top: 0 !important; }
</style>
""", unsafe_allow_html=True)

st.title("💧 Otter Hydraulic System Dashboard")

import dashboard_app_v22_wrapped as dash_main
import dashboard_cluster_app_v4_wrapped_v2 as dash_cluster

tab1, tab2 = st.tabs(["System Dashboard", "Clustering Status Dashboard"])

with tab1:
    dash_main.main()

with tab2:
    dash_cluster.main()

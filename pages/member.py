import streamlit as st

st.set_page_config(
    page_title="Project Members",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    [data-testid="stSidebarNav"] {
        display: none;
    }

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #F7F9FC;
    }

    [data-testid="stHeader"] {
        background: rgba(247, 249, 252, 0.85);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #182538 0%, #21324A 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #F5F7FB !important;
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    .member-hero {
        background: linear-gradient(135deg, #FFFFFF 0%, #EEF6FF 100%);
        border: 1px solid #E4E7EC;
        border-radius: 24px;
        padding: 30px 32px;
        box-shadow: 0 12px 28px rgba(16, 24, 40, 0.06);
        margin-bottom: 26px;
    }

    .section-title {
        font-size: 44px;
        font-weight: 800;
        color: #23324A;
        letter-spacing: -1.2px;
        margin-bottom: 8px;
    }

    .section-caption {
        color: #697586;
        font-size: 17px;
        line-height: 1.7;
        max-width: 900px;
    }

    /* Style untuk card container Streamlit */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-radius: 22px;
        box-shadow: 0 8px 24px rgba(16, 24, 40, 0.06);
        border: 1px solid #E4E7EC;
        padding: 8px;
    }

    /* Style gambar */
    img {
        border-radius: 16px;
        height: 260px;
        object-fit: cover;
        object-position: center;
    }

    h3 {
        color: #23324A !important;
        font-size: 20px !important;
        font-weight: 800 !important;
        margin-top: 12px !important;
        margin-bottom: 4px !important;
    }

    p {
        color: #5F6B7A;
        font-size: 15px;
        line-height: 1.6;
    }
            /* Tombol Member di sidebar */
section[data-testid="stSidebar"] .stButton > button {
    background-color: #34465F !important;
    color: #F5F7FB !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #4A6080 !important;
    border-color: rgba(255,255,255,0.3) !important;
}

/* Tombol Dashboard (disabled) */
section[data-testid="stSidebar"] .stButton > button:disabled {
    background-color: #1971C2 !important;
    color: white !important;
    opacity: 1 !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 📡 Menu")

    if st.button("📊 Dashboard", use_container_width=True):
        st.switch_page("streamlit_app.py")

    st.button("👥 Member", use_container_width=True, disabled=True)


# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="member-hero">
    <div class="section-title">👥 Project Members</div>
    <div class="section-caption">
        This page shows the members who worked on the Telco Customer Churn Dashboard project.
        Each member helped build the data analysis, machine learning model, and dashboard visualization.
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# MEMBER CARDS
# =========================================================
m1, m2, m3 = st.columns(3)

with m1:
    with st.container(border=True):
        st.image("pages/kevin f.JPG", use_container_width=True)
        st.markdown("### Kevin Febrian Setiadi")
        st.markdown("**0706022410001**")
        st.write("Worked on data preprocessing, data cleaning, and analysis flow.")

with m2:
    with st.container(border=True):
        st.image("pages/ethan.JPG", use_container_width=True)
        st.markdown("### Ethan Cannavaro Lauda")
        st.markdown("**0706022410002**")
        st.write("Worked on model training, model comparison, and evaluation results.")

with m3:
    with st.container(border=True):
        st.image("pages/casey daniella w.JPG", use_container_width=True)
        st.markdown("### Casey Daniella Winarto")
        st.markdown("**0706022410026**")
        st.write("Worked on dashboard UI/UX, visualization, and Streamlit implementation.")

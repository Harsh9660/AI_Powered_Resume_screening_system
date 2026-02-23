import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.processor import ResumeProcessor

@st.cache_resource
def load_processor():
    return ResumeProcessor()

processor = load_processor()

st.set_page_config(page_title="Resume Intelligence", page_icon="", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button {
        width: 100%; border-radius: 10px; height: 3em;
        background-color: #2e7d32; color: white;
        font-weight: bold; border: none; transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #388e3c;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .skill-badge {
        display: inline-block; padding: 4px 12px; margin: 4px;
        border-radius: 20px; background-color: #2e7d32;
        color: white; font-size: 0.8em;
    }
    .signal-box {
        background-color: #1e2130; border-radius: 12px;
        padding: 14px; border: 1px solid #30363d; margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Resume Intelligence System")
st.markdown("Match candidates against job requirements with **transparent AI scoring** across 3 signals.")


with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("Backend API Endpoint", "http://localhost:8000/rank-resumes")
    st.divider()
    st.markdown("""
    **Scoring Signals:**
    - 🎯 **Skill Match** (40%) — exact skill keywords
    - 📄 **Text Similarity** (35%) — TF-IDF cosine match
    - 🔑 **Keyword Coverage** (25%) — JD word presence
    """)
    st.info("Ensure the FastAPI server is running before processing.")


col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.header("📄 Job Requirements")
    jd_text = st.text_area("Job Description:", height=200,
                           placeholder="Paste JD here — mention skills like Python, AWS, Docker, Machine Learning...")

    if jd_text:
        jd_skills_preview = processor.extract_skills(jd_text)
        if jd_skills_preview:
            st.markdown("**Detected Skills in JD:**")
            badges = "".join([
                f'<span class="skill-badge" style="background-color:#1565c0;">{s}</span>'
                for s in sorted(jd_skills_preview)
            ])
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No standard tech skills detected. Keyword matching may be limited.")

    st.header("📤 Upload Candidate's Resume")
    uploaded_files = st.file_uploader("Select PDF Resumes:", type=["pdf"], accept_multiple_files=True)

    if st.button("🚀 Run Matching Engine"):
        if not jd_text or not uploaded_files:
            st.error("Please provide both a Job Description and at least one Resume.")
        else:
            st.session_state.process = True
            st.session_state.jd_text = jd_text
            st.session_state.uploaded_files = uploaded_files


if st.session_state.get("process"):
    with st.spinner("🔍 Analyzing candidate profiles..."):
        try:
            jd_txt = st.session_state.jd_text
            files_to_send = [
                ("resumes", (f.name, f, "application/pdf"))
                for f in st.session_state.uploaded_files
            ]
            response = requests.post(api_url, data={"job_description": jd_txt}, files=files_to_send)

            if response.status_code == 200:
                results = response.json().get("ranked_resumes", [])

                if results:
                    with col2:
                        st.header("📊 Ranking Insights")

                        
                        m1, m2, m3 = st.columns(3)
                        avg_score = sum(r["score"] for r in results) / len(results)
                        m1.metric("Total Resumes", len(results))
                        m2.metric("Avg Match", f"{avg_score*100:.1f}%")
                        #m3.metric("Top Score", f"{results[0]['score']*100:.1f}%")

                        st.divider()
 
                        for idx, res in enumerate(results):
                            score_pct = round(res["score"] * 100, 1)
                            bar_color = "green" if score_pct >= 60 else ("orange" if score_pct >= 35 else "red")

                            with st.expander(
                                f"**#{idx+1}  {res['filename']}** — Match: {score_pct}%",
                                expanded=(idx == 0)
                            ):
                                c_gauge, c_detail = st.columns([1, 2])

                                with c_gauge:
                                    fig = go.Figure(go.Indicator(
                                        mode="gauge+number",
                                        value=score_pct,
                                        number={"suffix": "%"},
                                        title={"text": "Match Score"},
                                        gauge={
                                            "axis": {"range": [0, 100]},
                                            "bar": {"color": bar_color},
                                            "steps": [
                                                {"range": [0, 35], "color": "#2d0000"},
                                                {"range": [35, 60], "color": "#2d2200"},
                                                {"range": [60, 100], "color": "#002d00"},
                                            ],
                                        }
                                    ))
                                    fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
                                    st.plotly_chart(fig, use_container_width=True)

                                with c_detail:
                                    # Skills found
                                    if res["skills"]:
                                        st.markdown("**Found Skills**")
                                        skill_html = "".join(
                                            [f'<span class="skill-badge">{s}</span>' for s in res["skills"]]
                                        )
                                        st.markdown(skill_html, unsafe_allow_html=True)
                                    else:
                                        st.warning("No known skills detected in resume.")

                                    # 3-signal breakdown bar chart
                                    if res.get("explanation"):
                                        exp = res["explanation"]
                                        sig_df = pd.DataFrame({
                                            "Signal": ["🎯 Skill Match", "📄 Text Similarity", "🔑 Keyword Coverage"],
                                            "Contribution (%)": [
                                                round(exp.get("skill_overlap_impact", 0) * 100, 1),
                                                round(exp.get("cosine_similarity_impact", 0) * 100, 1),
                                                round(exp.get("keyword_density_impact", 0) * 100, 1),
                                            ]
                                        })
                                        fig2 = px.bar(
                                            sig_df, x="Contribution (%)", y="Signal",
                                            orientation="h", title="Score Breakdown",
                                            color="Contribution (%)",
                                            color_continuous_scale=["#c62828", "#f9a825", "#2e7d32"],
                                            range_color=[0, 40]
                                        )
                                        fig2.update_layout(
                                            showlegend=False, coloraxis_showscale=False,
                                            height=220, margin=dict(l=10, r=10, t=35, b=10)
                                        )
                                        st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("No results returned from the API.")
            else:
                st.error(f"API Error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to API. Make sure `python3 src/api/main.py` is running.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

    st.session_state.process = False


st.divider()
st.caption("AI-Powered Resume Screening System · Formula-based scoring: Skill Match 40% + Text Similarity 35% + Keyword Density 25%")

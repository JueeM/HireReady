import streamlit as st
import sys
import os
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag"))

from loader import load_resume
from chunker import chunk_text
from embedder import embed_and_store
from pipeline import run_pipeline

st.set_page_config(page_title="HireReady", page_icon="💼", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.resume_processed = False

collection_name = f"resume_{st.session_state.session_id}"

st.title("💼 HireReady")
st.subheader("AI-powered job application assistant")

uploaded_file = st.file_uploader("📄 Upload your resume (PDF)", type=["pdf"])

if uploaded_file is not None and not st.session_state.resume_processed:
    with st.spinner("Processing your resume..."):
        temp_path = f"temp_resume_{st.session_state.session_id}.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        text = load_resume(temp_path)
        chunks = chunk_text(text)
        embed_and_store(chunks, collection_name=collection_name)

        os.remove(temp_path)
        st.session_state.resume_processed = True

    st.success(f"Resume processed — {len(chunks)} sections indexed")

if not st.session_state.resume_processed:
    st.info("Upload your resume above to get started")
    st.stop()

st.markdown("Paste a job description and get a tailored resume + cover letter grounded in your actual experience.")
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    job_description = st.text_area("📋 Paste Job Description here", height=300)

with col2:
    company_name = st.text_input("🏢 Company Name")
    st.markdown("---")
    st.markdown("**Agents in this pipeline:**")
    st.markdown("1. 🔍 Gap Analyzer")
    st.markdown("2. ✍️ Tailoring Agent")
    st.markdown("3. 📊 Resume Critic (ATS score)")
    st.markdown("4. 📄 Cover Letter Agent")
    st.markdown("5. 🎤 Interview Question Generator")
    run_button = st.button("🚀 Generate Application", type="primary", use_container_width=True)

st.divider()

if run_button:
    if not job_description:
        st.error("Please paste a job description first.")
    elif not company_name:
        st.error("Please enter the company name.")
    else:
        with st.spinner("Running 5 agents... this takes ~20 seconds"):
            result = run_pipeline(job_description, company_name, collection_name)

        st.markdown("## 🔍 Gap Analysis")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("### ✅ You Have")
            for item in result["gap_report"].get("have", []):
                st.markdown(f"- {item}")
        with col_b:
            st.markdown("### ❌ Missing")
            for item in result["gap_report"].get("missing", []):
                st.markdown(f"- {item}")
        with col_c:
            st.markdown("### 🔄 Reposition")
            for item in result["gap_report"].get("reposition", []):
                st.markdown(f"- {item}")

        st.divider()

        st.markdown("## 📊 ATS Score")
        ats = result["ats_report"]
        score = ats.get("ats_score", 0)
        st.metric("ATS Compatibility Score", f"{score}/100")
        st.progress(score / 100)

        col_d, col_e = st.columns(2)
        with col_d:
            st.markdown("**Strengths**")
            for s in ats.get("strengths", []):
                st.markdown(f"- {s}")
        with col_e:
            st.markdown("**Quick fixes**")
            for f in ats.get("quick_fixes", []):
                st.markdown(f"- {f}")

        st.divider()

        st.markdown("## ✍️ Tailored Resume Bullets")
        bullets_text = "\n".join([f"• {b}" for b in result["tailored_bullets"]])
        st.text_area("", value=bullets_text, height=200)

        st.divider()

        st.markdown("## 📄 Cover Letter")
        st.text_area("", value=result["cover_letter"], height=300)

        st.divider()

        st.markdown("## 🎤 Likely Interview Questions")
        for i, q in enumerate(result["interview_questions"], 1):
            st.markdown(f"{i}. {q}")

        st.divider()

        full_output = f"""HIREREADY — APPLICATION PACKAGE
Company: {company_name}

ATS SCORE: {score}/100

TAILORED RESUME BULLETS
{chr(10).join(['• ' + b for b in result['tailored_bullets']])}

COVER LETTER
{result['cover_letter']}

LIKELY INTERVIEW QUESTIONS
{chr(10).join([f'{i+1}. {q}' for i, q in enumerate(result['interview_questions'])])}
"""
        st.download_button(
            label="⬇️ Download Full Application Package",
            data=full_output,
            file_name=f"HireReady_{company_name}.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.success("✅ Done!")
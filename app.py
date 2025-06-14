import streamlit as st
import joblib
import pandas as pd
from utils.pre import clean_text
from utils.parse_resume import extract_text_from_pdf, extract_text_from_docx
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os
from io import BytesIO
from fpdf import FPDF

def find_missing_skills(resume_text, required_skills):
    resume_tokens = set(resume_text.lower().split())
    job_skills = [skill.strip().lower() for skill in required_skills.split(',')]
    missing = [skill for skill in job_skills if skill not in resume_tokens]
    return missing

# Streamlit setup
st.set_page_config(page_title="AI Resume Screener + Job Matcher", layout="centered")
st.title("AI Resume Screener + ML Job Matcher")

# File uploader
uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    try:
        # Save file temporarily
        save_path = "uploaded_resumes"
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_file_path = os.path.join(save_path, f"{timestamp}_{uploaded_file.name}")

        with open(saved_file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Extract and clean resume text
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "pdf":
            resume_text = extract_text_from_pdf(saved_file_path)
        else:
            resume_text = extract_text_from_docx(saved_file_path)

        clean_resume = clean_text(resume_text)

        # Show extracted text
        st.subheader("Extracted Resume Text")
        st.text_area("Cleaned Text from Resume:", clean_resume, height=200)

        # Load model and vectorizer
        try:
            vectorizer = joblib.load("utils/vectorizer.pkl")
            job_vectors = joblib.load("model/matcher_model.pkl")
        except FileNotFoundError:
            st.error("Model files not found. Please run `train_model.py` first.")
            st.stop()

        # Load job descriptions and compute scores
        job_df = pd.read_csv("model/cleaned_jobs.csv")
        X_resume = vectorizer.transform([clean_resume])
        scores = cosine_similarity(X_resume, job_vectors)[0]
        job_df["Score"] = scores
        top_jobs = job_df.sort_values(by="Score", ascending=False).head(5)

        # Display matches
        st.subheader("Top Matching Jobs")
        for _, row in top_jobs.iterrows():
            st.markdown(f"### {row['job_title']} - Match Score: `{row['Score']:.2f}`")
            st.markdown(f"**Required Skills:** {row['required_skills']}")
            st.markdown(f"**Description:** {row['description']}")

            missing_skills = find_missing_skills(clean_resume, row['required_skills'])
            if missing_skills:
                st.markdown(f"**Missing Skills:** {', '.join(missing_skills)}")
            else:
                st.markdown("✅ **All Required Skills Present!**")
            st.markdown("---")

        # --- Create and write to PDF ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(0, 10, "Top Matching Jobs", ln=True, align="C")
        pdf.ln(10)

        for _, row in top_jobs.iterrows():
            pdf.set_font("Arial", 'B', size=12)
            pdf.multi_cell(0, 10, f"{row['job_title']} - Match Score: {row['Score']:.2f}")

            pdf.set_font("Arial", size=11)
            pdf.multi_cell(0, 8, f"Required Skills: {row['required_skills']}")
            pdf.multi_cell(0, 8, f"Description: {row['description']}")

            missing_skills = find_missing_skills(clean_resume, row['required_skills'])
            if missing_skills:
                pdf.multi_cell(0, 8, f"Missing Skills: {', '.join(missing_skills)}")
            else:
                pdf.multi_cell(0, 8, "All Required Skills Present!")
            pdf.ln(5)

        # --- Convert PDF to byte stream (correct version) ---
        pdf_bytes = pdf.output(dest='S').encode('latin1')  # S returns as string, encode it to bytes
        pdf_output = BytesIO(pdf_bytes)

        # --- Download Button ---
        st.download_button(
            label="📥 Download Top Matches as PDF",
            data=pdf_output,
            file_name="top_matched_jobs.pdf",
            mime="application/pdf"
        )


    except Exception as e:
        st.error(f"❌ Error processing the resume: {str(e)}")

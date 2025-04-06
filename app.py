import streamlit as st
import PyPDF2
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------------------
# Function to extract text from PDF
# -------------------------------
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# -------------------------------
# Text Cleaning Function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    return text

# -------------------------------
# Extract Keywords using TF-IDF
# -------------------------------
def extract_keywords(text, top_n=20):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([text])
    keywords_df = pd.DataFrame(tfidf.T.todense(), index=vectorizer.get_feature_names_out(), columns=["tfidf"])
    top_keywords = keywords_df.sort_values(by="tfidf", ascending=False).head(top_n)
    return list(top_keywords.index)

# -------------------------------
# Match Keywords and Generate Score
# -------------------------------
def compare_resume_with_job(resume_keywords, job_keywords):
    matched = list(set(resume_keywords) & set(job_keywords))
    missing = list(set(job_keywords) - set(resume_keywords))
    match_score = round((len(matched) / len(job_keywords)) * 100, 2) if job_keywords else 0
    return match_score, matched, missing

# -------------------------------
# Downloadable Report
# -------------------------------
def create_report(job_title, score, matched, missing):
    report = f"Job Role: {job_title}\n"
    report += f"Match Score: {score}%\n\n"
    report += "✅ Keywords Present in Resume:\n"
    for word in matched:
        report += f"- {word}\n"
    report += "\n❌ Missing Keywords from Resume:\n"
    for word in missing:
        report += f"- {word}\n"
    return report

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="centered")
st.title("📄 AI Resume Analyzer (Offline)")
st.write("Analyze your resume based on the job description and get suggestions instantly — fully offline!")

# Inputs
job_title = st.text_input("🔍 Enter Job Title", "")
job_description = st.text_area("📋 Paste Job Description", height=200)
resume_file = st.file_uploader("📤 Upload Your Resume (PDF Only)", type=["pdf"])

# Analyze Button
if st.button("Analyze Resume") and resume_file and job_description:
    with st.spinner("Analyzing resume..."):
        resume_text = extract_text_from_pdf(resume_file)
        clean_resume = clean_text(resume_text)
        clean_job_desc = clean_text(job_description)

        resume_keywords = extract_keywords(clean_resume, top_n=30)
        job_keywords = extract_keywords(clean_job_desc, top_n=20)

        score, matched_keywords, missing_keywords = compare_resume_with_job(resume_keywords, job_keywords)

        # Results
        st.subheader("📊 Match Results")
        st.metric(label="Match Score", value=f"{score}%")
        st.success(f"✅ Keywords Found in Resume: {', '.join(matched_keywords)}")
        st.warning(f"❌ Missing Keywords: {', '.join(missing_keywords)}")

        # Chart
        st.subheader("📈 Keyword Match Overview")
        fig, ax = plt.subplots()
        ax.bar(["Matched", "Missing"], [len(matched_keywords), len(missing_keywords)], color=["green", "red"])
        ax.set_ylabel("Number of Keywords")
        st.pyplot(fig)

        # Download Report
        st.subheader("📥 Download Suggestions Report")
        report_text = create_report(job_title, score, matched_keywords, missing_keywords)
        buffer = BytesIO()
        buffer.write(report_text.encode())
        buffer.seek(0)
        st.download_button(label="Download Report (.txt)", data=buffer, file_name="resume_analysis.txt", mime="text/plain")

elif st.button("Analyze Resume"):
    st.warning("Please provide both a job description and upload a resume.")

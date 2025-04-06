import streamlit as st
import PyPDF2
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------------------
# Extract text from uploaded PDF
# -------------------------------
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# -------------------------------
# Clean text
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)      # remove numbers
    return text

# -------------------------------
# Extract top N keywords using TF-IDF
# -------------------------------
def extract_keywords(text, top_n=20):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([text])
    keywords_df = pd.DataFrame(tfidf.T.todense(), index=vectorizer.get_feature_names_out(), columns=["tfidf"])
    top_keywords = keywords_df.sort_values(by="tfidf", ascending=False).head(top_n)
    return list(top_keywords.index)

# -------------------------------
# Compare job vs resume
# -------------------------------
def compare_resume_with_job(resume_keywords, job_keywords):
    matched = list(set(resume_keywords) & set(job_keywords))
    missing = list(set(job_keywords) - set(resume_keywords))
    match_score = round((len(matched) / len(job_keywords)) * 100, 2) if job_keywords else 0
    return match_score, matched, missing

# -------------------------------
# Generate Downloadable Report
# -------------------------------
def create_report(job_title, score, matched, missing, projects, tech_stack):
    report = f"Job Role: {job_title}\n"
    report += f"Match Score: {score}%\n\n"
    report += "✅ Keywords Found in Resume:\n"
    for word in matched:
        report += f"- {word}\n"
    report += "\n❌ Missing Keywords (Consider Adding):\n"
    for word in missing:
        report += f"- {word}\n"
    report += "\n🧠 Detected Tech Stack:\n"
    for tech in tech_stack:
        report += f"- {tech}\n"
    report += "\n📂 Detected Projects:\n"
    for proj in projects:
        report += f"- {proj}\n"
    return report

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="centered")
st.title("📄 AI Resume Analyzer (Offline & Free)")
st.write("Upload your resume, provide a job title and description — get smart analysis & suggestions!")

# Inputs
job_title = st.text_input("🧑‍💼 Job Title", "")
job_description = st.text_area("📝 Job Description", height=200)
resume_file = st.file_uploader("📤 Upload Resume (PDF)", type=["pdf"])

# Button
analyze_clicked = st.button("🔍 Analyze Resume")

# Logic
if analyze_clicked:
    if resume_file and job_description:
        with st.spinner("Analyzing..."):
            # Extract and clean text
            resume_text = extract_text_from_pdf(resume_file)
            clean_resume = clean_text(resume_text)
            clean_job_desc = clean_text(job_description)

            # Keywords
            resume_keywords = extract_keywords(clean_resume, top_n=30)
            job_keywords = extract_keywords(clean_job_desc, top_n=20)

            score, matched_keywords, missing_keywords = compare_resume_with_job(resume_keywords, job_keywords)

            # Detect projects
            project_lines = [line.strip() for line in resume_text.split("\n") if re.search(r'\b(project|developed|built|implemented|created)\b', line, re.IGNORECASE)]

            # Detect tech stack
            tech_terms = ["python", "java", "flask", "django", "ml", "ai", "react", "streamlit", "pandas", "numpy", "nlp", "cv", "sklearn", "gpt", "bert"]
            tech_stack = [term for term in tech_terms if re.search(term, resume_text, re.IGNORECASE)]

            # Match results
            st.subheader("📊 Match Results")
            st.metric(label="Match Score", value=f"{score}%")
            st.success(f"✅ Found Keywords: {', '.join(matched_keywords)}")
            st.warning(f"❌ Missing Keywords: {', '.join(missing_keywords)}")

            # Projects
            st.subheader("📂 Detected Projects in Resume")
            if project_lines:
                for proj in project_lines:
                    st.markdown(f"- {proj}")
            else:
                st.info("No clear project section found. Make sure your resume has project titles with brief descriptions.")

            # Tech stack
            st.subheader("🧠 Tech Stack Mentioned")
            if tech_stack:
                st.write(", ".join(set(tech_stack)))
            else:
                st.info("Tech stack not detected. Mention technologies clearly in skills or project sections.")

            # Suggestions
            st.subheader("📌 Suggestions Based on Analysis")
            if missing_keywords:
                for kw in missing_keywords:
                    st.markdown(f"- Try adding something about **{kw}** in your skills or project section.")
            else:
                st.markdown("✅ Your resume already covers all the important keywords!")

            # Chart
            st.subheader("📈 Keyword Match Overview")
            fig, ax = plt.subplots()
            ax.bar(["Matched", "Missing"], [len(matched_keywords), len(missing_keywords)], color=["green", "red"])
            ax.set_ylabel("Number of Keywords")
            st.pyplot(fig)

            # Report
            st.subheader("📥 Download Text Report")
            report_text = create_report(job_title, score, matched_keywords, missing_keywords, project_lines, tech_stack)
            buffer = BytesIO()
            buffer.write(report_text.encode())
            buffer.seek(0)
            st.download_button(label="Download Report (.txt)", data=buffer, file_name="resume_analysis.txt", mime="text/plain")
    else:
        st.warning("⚠️ Please upload a resume and enter job details.")

from flask import Flask, render_template, request
import docx2txt
import PyPDF2

app = Flask(__name__)

job_keywords = [
    "python", "machine learning", "data analysis", "flask", "api",
    "pandas", "tensorflow", "nlp", "deep learning", "cloud"
]

def extract_text(file):
    text = ""
    if file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    elif file.filename.endswith(".docx"):
        text = docx2txt.process(file)
    return text.lower()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        resume = request.files["resume"]
        if resume:
            text = extract_text(resume)
            matched = [kw for kw in job_keywords if kw in text]
            percent = round((len(matched) / len(job_keywords)) * 100, 2)
            suggestions = list(set(job_keywords) - set(matched))
            return render_template("index.html", match=percent, suggestions=suggestions, submitted=True)
    return render_template("index.html", submitted=False)

if __name__ == "__main__":
    app.run(debug=True)

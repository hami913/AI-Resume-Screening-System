import os
import re
import joblib
import numpy as np
import pandas as pd

from pypdf import PdfReader
from docx import Document


# =========================
# Paths
# =========================

MODEL_PATH = "models/best_pipeline.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# =========================
# Load trained model
# =========================

pipeline = joblib.load(MODEL_PATH)


# =========================
# Text Extraction
# =========================

def extract_text_from_pdf(file_path):
    """Extract text from a PDF resume."""
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text.strip()


def extract_text_from_docx(file_path):
    """Extract text from a DOCX resume."""
    document = Document(file_path)

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    return text.strip()


def extract_resume_text(file_path):
    """Automatically extract resume text based on file type."""

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)

    elif extension == ".docx":
        return extract_text_from_docx(file_path)

    else:
        raise ValueError(
            "Unsupported file type. Please provide a PDF or DOCX resume."
        )


# =========================
# Basic Feature Extraction
# =========================

def clean_resume(text):
    """Basic resume text cleaning."""

    text = text.lower()

    text = re.sub(r"\S+@\S+", " ", text)

    text = re.sub(
        r"\+?\d[\d\s().-]{7,}\d",
        " ",
        text
    )

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_skills(text):
    """Extract recognized skills from resume text."""

    skill_keywords = [
        "python",
        "java",
        "c++",
        "javascript",
        "typescript",
        "sql",
        "html",
        "css",
        "react",
        "angular",
        "node",
        "django",
        "flask",
        "tensorflow",
        "pytorch",
        "keras",
        "scikit-learn",
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "data science",
        "data analysis",
        "nlp",
        "computer vision",
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "git",
        "github",
        "firebase",
        "flutter",
        "dart",
        "mongodb",
        "mysql",
        "postgresql",
        "power bi",
        "tableau",
        "excel",
        "statistics",
    ]

    text_lower = text.lower()

    return [
        skill
        for skill in skill_keywords
        if skill in text_lower
    ]


def extract_features(text):
    """Extract the numerical features used by the trained pipeline."""

    cleaned = clean_resume(text)

    words = cleaned.split()

    word_count = len(words)

    unique_word_count = len(set(words))

    resume_length = len(cleaned)

    if word_count > 0:
        avg_word_length = sum(len(word) for word in words) / word_count
    else:
        avg_word_length = 0

    email_present = int(
        bool(re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.I))
    )

    phone_present = int(
        bool(re.search(r"\+?\d[\d\s().-]{7,}\d", text))
    )

    skill_count = len(extract_skills(text))

    return {
        "clean_resume": cleaned,
        "resume_length": resume_length,
        "word_count": word_count,
        "unique_word_count": unique_word_count,
        "avg_word_length": avg_word_length,
        "email_present": email_present,
        "phone_present": phone_present,
        "skill_count": skill_count,
    }


# =========================
# Skill Normalization
# =========================

SKILL_ALIASES = {
    "apache spark": "spark",
    "spark framework": "spark",
    "scikit learn": "scikit-learn",
    "scikit learn library": "scikit-learn",
    "machine-learning": "machine learning",
    "deep-learning": "deep learning",
    "artificial-intelligence": "artificial intelligence",
    "natural language processing": "nlp",
    "postgres": "postgresql",
    "ms sql": "sql",
    "microsoft sql server": "sql",
    "github": "github",
    "git": "git",
    "sql server": "sql",
    "powerbi": "power bi",
    "problem-solving": "problem solving",
}

# Conservative related-skill relationships.
# Related matches receive less weight than exact matches.
SKILL_FAMILIES = {
    "machine learning": {"deep learning", "artificial intelligence", "tensorflow", "pytorch"},
    "deep learning": {"machine learning", "artificial intelligence", "tensorflow", "pytorch"},
    "artificial intelligence": {"machine learning", "deep learning", "tensorflow", "pytorch"},
    "data science": {"data analysis", "statistical analysis", "statistics"},
    "data analysis": {"data science", "statistical analysis", "statistics"},
    "sql": {"sql databases", "sql knowledge", "sql queries", "sql scripting"},
    "python": {"scripting languages", "programming languages"},
    "spark": {"stream analytics"},
    "cloud computing": {"cloud technologies"},
    "cloud technologies": {"cloud computing"},
    "data modeling": {"data management", "data warehousing"},
    "data warehousing": {"data management", "data modeling", "etl"},
    "git": {"version control systems"},
    "github": {"version control systems"},
}

RELATED_SKILL_WEIGHT = 0.50

def normalize_skill(skill):
    """Normalize skill names for reliable matching."""

    skill = str(skill).strip().lower()
    return SKILL_ALIASES.get(skill, skill)


def normalize_skills(skills):
    """Normalize and deduplicate a collection of skills."""

    return {
        normalize_skill(skill)
        for skill in skills
        if str(skill).strip()
    }


# =========================
# Career Prediction
# =========================

def predict_careers(text, top_n=5):
    """Predict the most suitable career categories."""

    features = extract_features(text)
    dataframe = pd.DataFrame([features])

    prediction_id = pipeline.predict(dataframe)[0]
    prediction = label_encoder.inverse_transform([int(prediction_id)])[0]

    decision_scores = pipeline.decision_function(dataframe)[0]
    class_ids = pipeline.classes_

    rankings = sorted(
        [
            (label_encoder.inverse_transform([int(class_id)])[0], score)
            for class_id, score in zip(class_ids, decision_scores)
        ],
        key=lambda x: x[1],
        reverse=True
    )

    skills = extract_skills(text)

    job_matches = match_jobs(skills, top_n=5)

    ats_score = (
        calculate_ats_score(features, job_matches[0])
        if job_matches
        else None
    )

    skill_gap = analyze_skill_gap(job_matches, skills)

    return {
        "predicted_career": prediction,
        "top_careers": rankings[:top_n],
        "skills": skills,
        "job_matches": job_matches,
        "ats_score": ats_score,
        "skill_gap": skill_gap,
        "features": features,
    }


# =========================
# Job Matching
# =========================

JOB_DATA_PATH = "data/all_job_post.csv"


def _parse_job_skills(value):
    """Parse the structured skill list stored in the job dataset."""

    import ast

    try:
        skills = ast.literal_eval(str(value))
        if isinstance(skills, list):
            return normalize_skills(skills)
    except (ValueError, SyntaxError):
        pass

    return set()


def match_jobs(resume_skills, top_n=5):
    """Match resume skills against jobs in the job dataset."""

    if not os.path.exists(JOB_DATA_PATH):
        raise FileNotFoundError(
            f"Job dataset not found: {JOB_DATA_PATH}"
        )

    jobs = pd.read_csv(JOB_DATA_PATH)

    resume_skills = normalize_skills(resume_skills)

    if not resume_skills:
        return []

    results = []

    for _, job in jobs.iterrows():

        job_skills = _parse_job_skills(job["job_skill_set"])

        if not job_skills:
            continue

        matched = set()
        missing = set()
        weighted_score = 0.0

        for job_skill in job_skills:
            if job_skill in resume_skills:
                matched.add(job_skill)
                weighted_score += 1.0
                continue

            related_resume_skill = None

            for resume_skill in resume_skills:
                if job_skill in SKILL_FAMILIES.get(resume_skill, set()):
                    related_resume_skill = resume_skill
                    break

            if related_resume_skill:
                matched.add(f"{job_skill} (~{related_resume_skill})")
                weighted_score += RELATED_SKILL_WEIGHT
            else:
                missing.add(job_skill)

        match_score = (
            weighted_score / len(job_skills) * 100
            if job_skills
            else 0
        )

        results.append({
            "job_id": job["job_id"],
            "category": job["category"],
            "job_title": job["job_title"],
            "match_score": round(match_score, 2),
            "matched_skills": sorted(matched),
            "missing_skills": sorted(missing),
        })

    results.sort(
        key=lambda item: (
            item["match_score"],
            len(item["matched_skills"])
        ),
        reverse=True
    )

    return results[:top_n]


# =========================
# ATS Scoring
# =========================

def calculate_ats_score(features, job_match):
    """Calculate an explainable ATS score for the best matching job."""

    skill_score = job_match["match_score"]

    contact_score = (
        features["email_present"] * 50
        + features["phone_present"] * 50
    )

    ats_score = (
        skill_score * 0.70
        + contact_score * 0.30
    )

    return {
        "ats_score": round(min(ats_score, 100), 2),
        "skill_coverage": round(skill_score, 2),
        "contact_score": round(contact_score, 2),
        "matched_skills": job_match["matched_skills"],
        "missing_skills": job_match["missing_skills"],
    }


# =========================
# Skill Gap Analysis
# =========================

def analyze_skill_gap(job_matches, resume_skills):
    """Analyze missing skills across the user's best job matches."""

    resume_skills = normalize_skills(resume_skills)

    if not job_matches:
        return {
            "total_missing_skills": 0,
            "critical_skills": [],
            "important_skills": [],
            "other_skills": [],
        }

    skill_frequency = {}

    for job in job_matches:
        for skill in job.get("missing_skills", []):
            normalized = normalize_skill(skill)

            if normalized not in resume_skills:
                skill_frequency[normalized] = (
                    skill_frequency.get(normalized, 0) + 1
                )

    total_jobs = len(job_matches)

    critical_skills = sorted(
        [
            skill for skill, count in skill_frequency.items()
            if count >= max(1, total_jobs * 0.6)
        ],
        key=lambda skill: (-skill_frequency[skill], skill)
    )

    important_skills = sorted(
        [
            skill for skill, count in skill_frequency.items()
            if skill not in critical_skills
            and count >= max(1, total_jobs * 0.3)
        ],
        key=lambda skill: (-skill_frequency[skill], skill)
    )

    other_skills = sorted(
        [
            skill for skill in skill_frequency
            if skill not in critical_skills
            and skill not in important_skills
        ],
        key=lambda skill: (-skill_frequency[skill], skill)
    )

    return {
        "total_missing_skills": len(skill_frequency),
        "critical_skills": critical_skills,
        "important_skills": important_skills,
        "other_skills": other_skills,
    }


# =========================
# Resume Analyzer
# =========================

def analyze_resume(file_path, top_n=5):
    """Complete resume analysis pipeline."""

    print("\n" + "=" * 60)
    print("AI RESUME ANALYZER")
    print("=" * 60)

    print(f"\nFile: {file_path}")

    print("\n[1/3] Extracting resume text...")

    text = extract_resume_text(file_path)

    if not text:
        raise ValueError(
            "No text could be extracted from this resume."
        )

    print(f"Extracted characters: {len(text):,}")

    print("\n[2/3] Running career prediction...")

    result = predict_careers(text, top_n=top_n)

    print("\n[3/3] Career analysis complete.")

    print("\n" + "-" * 60)
    print("PREDICTED CAREER")
    print("-" * 60)

    print(result["predicted_career"])

    print("\n" + "-" * 60)
    print(f"TOP {top_n} CAREER MATCHES")
    print("-" * 60)

    for rank, (career, score) in enumerate(
        result["top_careers"],
        start=1
    ):
        print(f"{rank}. {career:<30} Score: {score:.4f}")

    print("\n" + "-" * 60)
    print("EXTRACTED SKILLS")
    print("-" * 60)

    print(f"Total Skills: {len(result['skills'])}")
    print(", ".join(sorted(result["skills"])))

    print("\n" + "-" * 60)
    print("ATS SCORE")
    print("-" * 60)

    if result["ats_score"]:
        ats = result["ats_score"]
        print(f"ATS Score:       {ats['ats_score']:.2f}%")
        print(f"Skill Coverage:  {ats['skill_coverage']:.2f}%")
        print(f"Contact Score:   {ats['contact_score']:.2f}%")
    else:
        print("ATS Score: N/A - no job matches found")

    print("\n" + "-" * 60)
    print("BEST JOB MATCH")
    print("-" * 60)

    if result["job_matches"]:
        best_job = result["job_matches"][0]

        print(f"Job Title:       {best_job['job_title']}")
        print(f"Category:        {best_job['category']}")
        print(f"Match Score:     {best_job['match_score']:.2f}%")

        print("\nMatched Skills:")
        print(", ".join(best_job["matched_skills"]) or "None")

        print("\nMissing Skills:")
        print(", ".join(best_job["missing_skills"]) or "None")
    else:
        print("No suitable job matches found.")

    print("\n" + "-" * 60)
    print("SKILL GAP ANALYSIS")
    print("-" * 60)

    gap = result["skill_gap"]

    print(f"Total Missing Skills: {gap['total_missing_skills']}")

    print("\nCritical Skills:")
    print(", ".join(gap["critical_skills"]) or "None")

    print("\nImportant Skills:")
    print(", ".join(gap["important_skills"]) or "None")

    print("\nOther Skills:")
    print(", ".join(gap["other_skills"]) or "None")

    print("\n" + "=" * 60)

    return result


# =========================
# Command Line Interface
# =========================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("python resume_analyzer.py <resume.pdf>")
        print("python resume_analyzer.py <resume.docx>")
        sys.exit(1)

    resume_path = sys.argv[1]

    if not os.path.exists(resume_path):
        print(f"\nError: File not found: {resume_path}")
        sys.exit(1)

    analyze_resume(resume_path)

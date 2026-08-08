import re
from typing import Dict, Any, List, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Default technical skill pattern for keyword extraction
DEFAULT_SKILL_PATTERN = r'\b(python|c\+\+|cpp|c#|sql|scikit-learn|xgboost|lightgbm|optuna|shap|lime|nlp|tfidf|pandas|numpy|matplotlib|seaborn|plotly|streamlit|fastapi|docker|git|github|mysql|postgresql|mongodb|machine learning|artificial intelligence|deep learning|data science|rest api|html|css|javascript|react)\b'

def clean_input_text(text: str) -> str:
    """
    Standardize and clean raw text for vectorization.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_skills(text: str, custom_pattern: str = DEFAULT_SKILL_PATTERN) -> Set[str]:
    """
    Extract technical skills from text matching the skill pattern.
    """
    cleaned = clean_input_text(text)
    matched = re.findall(custom_pattern, cleaned)
    return set(matched)

def calculate_ats_match(resume_text: str, job_description: str) -> Dict[str, Any]:
    """
    Compute ATS score via Cosine Similarity and perform skill gap analysis.
    """
    cleaned_resume = clean_input_text(resume_text)
    cleaned_jd = clean_input_text(job_description)

    if not cleaned_resume or not cleaned_jd:
        return {
            "ats_score": 0.0,
            "matched_skills": [],
            "missing_skills": [],
            "status": "Error: Empty text provided."
        }

    # 1. Cosine Similarity via TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([cleaned_resume, cleaned_jd])
    
    # Calculate similarity score (0.0 to 1.0) -> scaled to percentage
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    raw_similarity = float(similarity_matrix[0][0])
    ats_score = round(raw_similarity * 100, 2)

    # 2. Skill Gap Extraction
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(job_description)

    matched_skills = list(resume_skills.intersection(jd_skills))
    missing_skills = list(jd_skills.difference(resume_skills))

    return {
        "ats_score": ats_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "resume_skill_count": len(resume_skills),
        "jd_skill_count": len(jd_skills),
        "status": "Success"
    }

if __name__ == "__main__":
    # Quick sanity test execution
    sample_resume = "Data Scientist experienced in Python, Machine Learning, Scikit-Learn, XGBoost, and SQL."
    sample_jd = "Looking for a Data Scientist proficient in Python, SQL, Docker, FastAPI, and Scikit-Learn."

    results = calculate_ats_match(sample_resume, sample_jd)
    print("--- ATS Test Run ---")
    print(f"ATS Match Score: {results['ats_score']}%")
    print(f"Matched Skills: {results['matched_skills']}")
    print(f"Missing Skills: {results['missing_skills']}")
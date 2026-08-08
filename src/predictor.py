import re
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
POSSIBLE_DIRS = [BASE_DIR / "models", BASE_DIR / "artifacts", BASE_DIR]

SKILL_PATTERN = r'(python|c\+\+|cpp|c#|sql|scikitlearn|xgboost|lightgbm|random forest|logistic regression|linear svm|nlp|tfidf|pandas|numpy|matplotlib|plotly|streamlit|fastapi|docker|mysql|postgresql|firebase|git|github|machine learning|artificial intelligence|data science|shap|lime|html|css|javascript|react)'

pipeline = None
label_encoder = None

# Asset Loading
for d in POSSIBLE_DIRS:
    for model_name in ["pipeline.pkl", "best_pipeline.pkl", "resume_classifier.pkl", "model.pkl"]:
        model_path = d / model_name
        if model_path.exists():
            pipeline = joblib.load(model_path)
            break
    if pipeline:
        break

for d in POSSIBLE_DIRS:
    for le_name in ["label_encoder.pkl", "encoder.pkl", "target_encoder.pkl"]:
        le_path = d / le_name
        if le_path.exists():
            label_encoder = joblib.load(le_path)
            break
    if label_encoder:
        break


def predict_resume_category(raw_resume: str) -> Dict[str, Any]:
    if not pipeline:
        raise RuntimeError("Model pipeline not loaded. Please check your models/ directory.")

    if not raw_resume.strip():
        raise ValueError("Resume text cannot be empty.")

    clean_text = re.sub(r'http\S+\s*', ' ', raw_resume)
    clean_text = re.sub(r'[^\w\s]', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).lower()

    words = clean_text.split()
    matched_skills = re.findall(SKILL_PATTERN, clean_text)

    feature_dict = {
        'clean_resume': clean_text,
        'resume_length': len(raw_resume),
        'word_count': len(words),
        'unique_word_count': len(set(words)),
        'avg_word_length': float(sum(len(w) for w in words) / max(len(words), 1)),
        'email_present': 1 if ("@" in raw_resume or "mailto" in clean_text) else 0,
        'phone_present': 1 if re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{10,}', raw_resume) else 0,
        'skill_count': len(matched_skills)
    }

    input_df = pd.DataFrame([feature_dict])
    raw_pred = pipeline.predict(input_df)[0]

    if label_encoder and hasattr(label_encoder, "inverse_transform"):
        category = str(label_encoder.inverse_transform([raw_pred])[0])
    elif hasattr(pipeline, "classes_") and isinstance(raw_pred, (int, np.integer)):
        category = str(pipeline.classes_[raw_pred])
    else:
        category = str(raw_pred)

    return {
        "predicted_category": category,
        "extracted_features": feature_dict,
        "matched_skills": list(set(matched_skills))
    }
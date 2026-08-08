import os
import joblib
import optuna
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

# Optuna logging verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# 1. SETUP MLFLOW EXPERIMENT
# ==========================================
mlflow.set_experiment("Resume_Classifier_Optimization")


# ==========================================
# 2. LOAD & PREPARE DATASET
# ==========================================
def load_data():
    # Adjust file path if your dataset is located elsewhere
    data_path = "data/processed_resumes.csv"
    if not os.path.exists(data_path):
        data_path = "processed_resumes.csv"

    df = pd.read_csv(data_path)
    
    # Ensure text column and target column exist
    X = df["clean_resume"].fillna("").astype(str)
    
    le = LabelEncoder()
    y = le.fit_transform(df["Category"])
    
    # Save Label Encoder artifact
    os.makedirs("models", exist_ok=True)
    joblib.dump(le, "models/label_encoder.pkl")
    
    return X, y, le

X, y, label_encoder = load_data()


# ==========================================
# 3. OPTUNA OBJECTIVE FUNCTION
# ==========================================
def objective(trial):
    # Select Classifier Architecture
    classifier_name = trial.suggest_categorical("classifier", ["LinearSVC", "LogisticRegression", "LightGBM"])
    
    # TF-IDF Vectorizer Hyperparameters
    max_features = trial.suggest_int("tfidf_max_features", 1000, 5000, step=1000)
    ngram_max = trial.suggest_int("tfidf_ngram_max", 1, 2)
    
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max), stop_words="english")
    
    # Model Selection & Hyperparameter Ranges
    if classifier_name == "LinearSVC":
        c_val = trial.suggest_float("svc_c", 0.01, 10.0, log=True)
        clf = LinearSVC(C=c_val, random_state=42, max_iter=2000)
        
    elif classifier_name == "LogisticRegression":
        c_val = trial.suggest_float("logreg_c", 0.01, 10.0, log=True)
        clf = LogisticRegression(C=c_val, max_iter=1000, random_state=42)
        
    elif classifier_name == "LightGBM":
        n_estimators = trial.suggest_int("lgb_n_estimators", 50, 200, step=50)
        learning_rate = trial.suggest_float("lgb_lr", 0.01, 0.2, log=True)
        num_leaves = trial.suggest_int("lgb_num_leaves", 20, 50)
        clf = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=42,
            verbosity=-1
        )

    pipeline = Pipeline([
        ("tfidf", vec),
        ("clf", clf)
    ])

    # 5-Fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    mean_accuracy = np.mean(scores)

    # Log Individual Trial to MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params(trial.params)
        mlflow.log_metric("cv_accuracy", mean_accuracy)
        mlflow.set_tag("model_type", classifier_name)

    return mean_accuracy


# ==========================================
# 4. RUN OPTUNA STUDY & SAVE BEST MODEL
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting Hyperparameter Optimization with Optuna & MLflow...")
    
    with mlflow.start_run(run_name="Optuna_Study_Parent"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)

        print("\n" + "="*50)
        print("🏆 BEST TRIAL RESULTS")
        print("="*50)
        print(f"Best Accuracy: {study.best_value * 100:.2f}%")
        print("Best Parameters:")
        for key, val in study.best_params.items():
            print(f"  - {key}: {val}")

        # Log Best Overall Parameters to Parent Run
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cv_accuracy", study.best_value)

        # Retrain Best Pipeline on Full Dataset
        best_params = study.best_params
        best_clf_name = best_params["classifier"]
        
        vec = TfidfVectorizer(
            max_features=best_params["tfidf_max_features"],
            ngram_range=(1, best_params["tfidf_ngram_max"]),
            stop_words="english"
        )
        
        if best_clf_name == "LinearSVC":
            clf = LinearSVC(C=best_params["svc_c"], random_state=42, max_iter=2000)
        elif best_clf_name == "LogisticRegression":
            clf = LogisticRegression(C=best_params["logreg_c"], max_iter=1000, random_state=42)
        elif best_clf_name == "LightGBM":
            clf = LGBMClassifier(
                n_estimators=best_params["lgb_n_estimators"],
                learning_rate=best_params["lgb_lr"],
                num_leaves=best_params["lgb_num_leaves"],
                random_state=42,
                verbosity=-1
            )

        best_pipeline = Pipeline([
            ("tfidf", vec),
            ("clf", clf)
        ])
        
        best_pipeline.fit(X, y)

        # Export Production Models
        joblib.dump(best_pipeline, "models/best_pipeline.pkl")
        joblib.dump(best_pipeline, "models/pipeline.pkl")
        joblib.dump(vec, "models/tfidf_vectorizer.pkl")

        # Log Final Artifacts to MLflow
        mlflow.sklearn.log_model(best_pipeline, "best_model_artifact")
        print("\n✅ Production model exported successfully to 'models/pipeline.pkl'!")
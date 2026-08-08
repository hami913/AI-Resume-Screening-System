import os
import sys
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

# Add src to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Ensure models directory exists
models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)

mlflow.set_experiment("Resume_Screener_Evaluation")

def train_and_track(X_train, y_train, preprocessor=None):
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(eval_metric='mlogloss'),
        "LightGBM": LGBMClassifier()
    }
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            if preprocessor:
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
            else:
                pipeline = model
                
            scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
            
            mlflow.log_param("model_name", name)
            mlflow.log_metric("cv_accuracy_mean", scores.mean())
            
            pipeline.fit(X_train, y_train)
            mlflow.sklearn.log_model(pipeline, f"model_{name}")
            
            # Save updated pipeline models locally
            if name == "XGBoost":
                pipe_path = os.path.join(models_dir, "pipeline.pkl")
                best_path = os.path.join(models_dir, "best_pipeline.pkl")
                joblib.dump(pipeline, pipe_path)
                joblib.dump(pipeline, best_path)
                print(f"✅ SUCCESSFULLY SAVED: {pipe_path}")

if __name__ == "__main__":
    try:
        from preprocessing import load_data_and_preprocessor
        X_train, y_train, preprocessor = load_data_and_preprocessor()
        train_and_track(X_train, y_train, preprocessor)
    except Exception as e:
        print(f"❌ Execution Error: {e}")
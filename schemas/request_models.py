from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


# -------------------------------------------------------------------
# Root / Health Check Schemas
# -------------------------------------------------------------------
class HealthCheckResponse(BaseModel):
    status: str = Field(..., example="healthy")
    app_name: str = Field(..., example="AI-Career-Assistant API")
    version: str = Field(..., example="1.0.0")


# -------------------------------------------------------------------
# Resume Classification (/predict)
# -------------------------------------------------------------------
class ResumeRequest(BaseModel):
    resume_text: str = Field(
        ...,
        min_length=10,
        description="Raw extracted text from the resume.",
        example="Experienced Data Scientist skilled in Python, Scikit-learn, Machine Learning, and NLP."
    )


class ResumePredictionResponse(BaseModel):
    predicted_category: str = Field(..., example="Data Science")
    extracted_features: Optional[Dict[str, Any]] = Field(
        None, 
        description="Dictionary of extracted structural and numerical text features."
    )
    matched_skills: Optional[List[str]] = Field(
        None, 
        example=["python", "machine learning", "nlp"]
    )


# -------------------------------------------------------------------
# ATS Match Score (/ats-match)
# -------------------------------------------------------------------
class ATSMatchRequest(BaseModel):
    resume_text: str = Field(
        ...,
        min_length=10,
        description="Raw or parsed resume text.",
        example="Senior Software Engineer with experience in Python, FastAPI, Docker, and PostgreSQL."
    )
    job_description: str = Field(
        ...,
        min_length=10,
        description="Target job description text.",
        example="Looking for a Python Developer proficient in FastAPI, Docker, and AWS."
    )


class ATSMatchResponse(BaseModel):
    ats_score: float = Field(..., example=85.5, description="Matching score percentage (0-100).")
    matching_skills: List[str] = Field(default_factory=list, example=["Python", "FastAPI", "Docker"])
    missing_skills: List[str] = Field(default_factory=list, example=["AWS", "PostgreSQL"])
    feedback: Optional[str] = Field(None, example="Strong technical alignment with target role.")


# -------------------------------------------------------------------
# Skill Extraction (/extract-skills)
# -------------------------------------------------------------------
class SkillExtractionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        description="Text content (resume or job description) to extract skills from.",
        example="Proficient in Python, TensorFlow, PyTorch, SQL, and Docker."
    )


class SkillExtractionResponse(BaseModel):
    extracted_skills: List[str] = Field(..., example=["Python", "TensorFlow", "PyTorch", "SQL", "Docker"])
    total_skills_found: int = Field(..., example=5)


# -------------------------------------------------------------------
# Job Recommendation (/recommend-jobs)
# -------------------------------------------------------------------
class JobRecommendationRequest(BaseModel):
    resume_text: str = Field(
        ...,
        min_length=10,
        description="Resume text used to query for relevant job matches."
    )
    top_n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top job recommendations to return."
    )


class RecommendedJob(BaseModel):
    job_title: str = Field(..., example="Machine Learning Engineer")
    similarity_score: float = Field(..., example=0.88)
    matched_skills: List[str] = Field(default_factory=list, example=["Python", "Scikit-Learn"])


class JobRecommendationResponse(BaseModel):
    recommendations: List[RecommendedJob]
    total_returned: int = Field(..., example=5)
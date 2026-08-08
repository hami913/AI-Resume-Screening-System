from fastapi import APIRouter, HTTPException, status
from schemas.request_models import (
    HealthCheckResponse,
    ResumeRequest,
    ResumePredictionResponse,
    ATSMatchRequest,
    ATSMatchResponse,
    SkillExtractionRequest,
    SkillExtractionResponse,
    JobRecommendationRequest,
    JobRecommendationResponse,
)

from src.predictor import predict_resume_category

try:
    from src.ats_score import calculate_ats_match, extract_skills
except ImportError:
    from ats_score import calculate_ats_match, extract_skills

try:
    from src.job_recommender import recommend_jobs
except ImportError:
    recommend_jobs = None


router = APIRouter()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="API Health Check",
    tags=["System"]
)
def health_check():
    return HealthCheckResponse(
        status="healthy",
        app_name="AI-Career-Assistant API",
        version="1.0.0"
    )


@router.post(
    "/predict",
    response_model=ResumePredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Resume Category",
    tags=["Resume Inference"]
)
def predict_category(payload: ResumeRequest):
    try:
        result = predict_resume_category(payload.resume_text)
        return ResumePredictionResponse(**result)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error predicting resume category: {str(e)}"
        )


@router.post(
    "/ats-match",
    response_model=ATSMatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Calculate ATS Match Score",
    tags=["ATS Analysis"]
)
def ats_match(payload: ATSMatchRequest):
    try:
        result = calculate_ats_match(payload.resume_text, payload.job_description)
        if isinstance(result, dict):
            return ATSMatchResponse(
                ats_score=result.get("ats_score", 0.0),
                matching_skills=result.get("matching_skills", []),
                missing_skills=result.get("missing_skills", []),
                feedback=result.get("feedback")
            )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating ATS match score: {str(e)}"
        )


@router.post(
    "/extract-skills",
    response_model=SkillExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract Skills from Text",
    tags=["Skill Processing"]
)
def extract_skills_endpoint(payload: SkillExtractionRequest):
    try:
        skills = list(extract_skills(payload.text))
        return SkillExtractionResponse(
            extracted_skills=skills,
            total_skills_found=len(skills)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting skills: {str(e)}"
        )


@router.post(
    "/recommend-jobs",
    response_model=JobRecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Recommend Matching Jobs",
    tags=["Job Matching"]
)
def recommend_jobs_endpoint(payload: JobRecommendationRequest):
    if not recommend_jobs:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Job recommendation module is not available in src."
        )
    try:
        recommendations = recommend_jobs(
            resume_text=payload.resume_text,
            top_n=payload.top_n
        )
        return JobRecommendationResponse(
            recommendations=recommendations,
            total_returned=len(recommendations)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error recommending jobs: {str(e)}"
        )
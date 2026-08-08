import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router

app = FastAPI(
    title="Enterprise AI Resume Screening API",
    description="REST API endpoints for resume classification, feature extraction, ATS scoring, and skill gap analysis.",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend requests (e.g., Streamlit / React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount centralized router
app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
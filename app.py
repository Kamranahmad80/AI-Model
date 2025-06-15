from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime
import tensorflow as tf
import numpy as np
import os
import logging
import time
import fastapi
from pathlib import Path

# Import dotenv for environment variables
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from services.resume_parser import ResumeParser
from services.job_matcher import JobMatcher
from services.huggingface_connector import HuggingFaceConnector
from database.db_connector import DatabaseConnector
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("job_recommendation_api")

# Define API models for request/response validation
class ExperienceItem(BaseModel):
    start_year: str
    end_year: str
    company: Optional[str] = ''
    role: Optional[str] = ''
    description: str

class EducationItem(BaseModel):
    degree: str
    institution: str
    start_year: str
    end_year: str

class ResumeData(BaseModel):
    skills: List[str]
    experience: List[ExperienceItem]
    education: Optional[List[EducationItem]] = None

class RecommendationRequest(BaseModel):
    user_id: str
    resume_data: ResumeData
    
    @validator('user_id')
    def user_id_must_be_valid(cls, v):
        from bson.objectid import ObjectId
        if not ObjectId.is_valid(v):
            raise ValueError('Invalid user_id format')
        return v

class JobFeedback(BaseModel):
    user_id: str
    job_id: str
    rating: int = Field(..., ge=1, le=5)  # Rating between 1-5

# Define API security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header == os.getenv("API_KEY"):
        return api_key_header
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Invalid API Key"
        )

# Initialize FastAPI app
app = FastAPI(
    title="Job Recommendation API",
    description="API for providing job recommendations based on resume analysis",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add middleware for logging requests and timing
@app.middleware("http")
async def add_process_time_header(request: fastapi.Request, call_next):
    start_time = time.time()
    try:
        # Log the incoming request
        logger.info(f"Request: {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
        
        # Process the request
        response = await call_next(request)
        
        # Add processing time header
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log the response status
        logger.info(f"Response: {response.status_code} in {process_time:.3f}s")
        
        return response
    except Exception as e:
        # Log any unhandled exceptions
        logger.error(f"Unhandled exception: {str(e)}")
        process_time = time.time() - start_time
        
        # Return a 500 response with error details
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "message": "Internal server error",
                "error": str(e),
                "process_time": f"{process_time:.3f}s"
            }
        )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,http://localhost:5000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time"]
)

# Initialize services
resume_parser = ResumeParser()
job_matcher = JobMatcher()
db = DatabaseConnector()

# Initialize services
try:
    # Initialize MongoDB connector
    db = DatabaseConnector()
    logger.info("Connected to MongoDB")
    
    # Initialize resume parser
    resume_parser = ResumeParser()
    logger.info("Initialized resume parser")
    
    # Initialize job matcher
    job_matcher = JobMatcher()
    logger.info("Initialized job matcher")
    
    # Initialize the recommendation model
    try:
        # Try to load local recommendation model first
        try:
            from recommender.model import RecommendationModel
            recommender = RecommendationModel()
            logger.info("Initialized local AI recommendation model")
            use_huggingface = False
        except Exception as e:
            logger.warning(f"Local model not available: {e}. Will use HuggingFace API instead.")
            # Initialize HuggingFace connector as fallback
            huggingface_connector = HuggingFaceConnector()
            logger.info("Initialized HuggingFace connector for model inference")
            use_huggingface = True
            recommender = None
        
        # We'll use recommender.model instead of model in the rest of the code
        model = None  # Not used anymore
    except Exception as e:
        logger.error(f"Error initializing recommendation models: {e}")
        recommender = None
        use_huggingface = False
except Exception as e:
    logger.error(f"Error initializing core services: {e}")
    raise

@app.get("/")
async def root():
    """Health check endpoint used by the JobFinder website to verify API availability"""
    return {"message": "Job Recommendation API", "status": "active", "version": "1.0.0", "docs": "/api/docs"}

@app.post("/api/analyze-resume", dependencies=[Depends(get_api_key)])
async def analyze_resume(file: UploadFile = File(...)):
    """
    Analyze uploaded resume and return skills and experience
    """
    try:
        # Log resume analysis request
        logger.info(f"Analyzing resume: {file.filename}")
        
        # Parse resume
        resume_text = await resume_parser.parse_resume(file)
        
        # Extract information
        skills = resume_parser.extract_skills(resume_text)
        experience = resume_parser.extract_experience(resume_text)
        education = resume_parser.extract_education(resume_text)
        
        logger.info(f"Successfully extracted {len(skills)} skills from resume")
        
        return {
            "success": True,
            "data": {
                "skills": skills,
                "experience": experience,
                "education": education
            }
        }
    except ValueError as e:
        logger.error(f"Value error in resume analysis: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing resume: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Error processing resume"
        )

@app.post("/api/recommend-jobs", dependencies=[Depends(get_api_key)])
async def recommend_jobs(request: RecommendationRequest):
    """
    Recommend jobs based on resume analysis
    """
    try:
        # Log recommendation request
        logger.info(f"Processing job recommendation for user: {request.user_id}")
        
        # Get user profile from database
        user_profile = await db.get_user_profile(request.user_id)
        if not user_profile:
            logger.warning(f"User profile not found: {request.user_id} - using resume data only")
            # Create a basic user profile from the resume data
            user_profile = {
                'id': request.user_id,
                'skills': request.resume_data.skills,
                'name': 'Anonymous User',  # Default name
                'email': '',  # Empty email
                'preferred_salary': 0,
                'preferred_location_code': 0,
                'preferred_location_text': '',
                'profile': {}
            }
        
        # Get matching jobs using our AI recommendation model
        try:
            # Prepare resume data dictionary
            resume_data_dict = request.resume_data.dict()
            
            # Make sure skills exist and are a list
            if 'skills' not in resume_data_dict or not resume_data_dict['skills']:
                resume_data_dict['skills'] = ["python", "javascript", "web development"]  # Default skills
                logger.warning(f"No skills found in resume data, adding default skills")
            
            # Make sure experience exists
            if 'experience' not in resume_data_dict or not resume_data_dict['experience']:
                resume_data_dict['experience'] = []
            
            # Make sure education exists
            if 'education' not in resume_data_dict or not resume_data_dict['education']:
                resume_data_dict['education'] = []
                
            # Decision tree for model selection:
            # 1. Try local model if available
            # 2. Try Hugging Face if available and local model isn't
            # 3. Fall back to rule-based matching as last resort
            
            if recommender:  # Local model is available
                logger.info("Using local AI model for recommendations")
                recommendations = recommender.get_recommendations(
                    resume_data_dict,
                    user_profile
                )
                logger.info(f"Generated {len(recommendations)} recommendations with local AI model")
                
            elif 'use_huggingface' in globals() and use_huggingface and 'huggingface_connector' in globals():  # Use Hugging Face
                logger.info("Using Hugging Face API for recommendations")
                # Get jobs from database to send to Hugging Face
                jobs_data = await db.get_jobs(limit=100)  # Limit to 100 jobs for API efficiency
                
                # Call Hugging Face API
                recommendations = huggingface_connector.get_recommendations(
                    resume_data_dict,
                    jobs_data,
                    user_profile
                )
                logger.info(f"Generated {len(recommendations)} recommendations with Hugging Face API")
                
            else:  # No AI models available
                logger.warning("No AI recommendation models available, using rule-based matching")
                recommendations = job_matcher.get_recommendations(
                    None,
                    resume_data_dict,
                    user_profile
                )
                logger.info(f"Generated {len(recommendations)} recommendations with rule-based matching")
            
            # If no recommendations, use rule-based matching as fallback
            if not recommendations or len(recommendations) == 0:
                logger.warning("AI model returned no recommendations, falling back to rule-based matching")
                recommendations = job_matcher.get_recommendations(
                    None,
                    request.resume_data.dict(),
                    user_profile
                )
                
        except Exception as e:
            # Log the error and use fallback for better user experience
            logger.error(f"Error using AI recommendation model: {e}")
            logger.warning("Falling back to rule-based matching due to error")
            recommendations = job_matcher.get_recommendations(
                None,
                request.resume_data.dict(),
                user_profile
            )
        
        logger.info(f"Generated {len(recommendations)} job recommendations")
        
        return {
            "success": True,
            "data": {
                "recommendations": recommendations
            }
        }
    except ValueError as e:
        logger.error(f"Value error in job recommendation: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Error generating recommendations"
        )

@app.post("/api/recommendation-feedback", dependencies=[Depends(get_api_key)])
async def submit_recommendation_feedback(feedback: JobFeedback, background_tasks: BackgroundTasks):
    """
    Submit user feedback about job recommendations
    """
    try:
        # Store feedback asynchronously
        background_tasks.add_task(
            db.store_recommendation_feedback,
            feedback.user_id,
            feedback.job_id,
            feedback.rating
        )
        
        logger.info(f"Recorded feedback: User {feedback.user_id}, Job {feedback.job_id}, Rating {feedback.rating}")
        
        return {"success": True, "message": "Feedback recorded successfully"}
    except Exception as e:
        logger.error(f"Error recording feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Error recording feedback"
        )

@app.post("/api/record-application", dependencies=[Depends(get_api_key)])
async def record_application(user_id: str, job_id: str, background_tasks: BackgroundTasks):
    """
    Record when a user applies to a job
    """
    try:
        # Record application asynchronously
        background_tasks.add_task(
            db.record_job_application,
            user_id,
            job_id
        )
        
        logger.info(f"Recorded job application: User {user_id}, Job {job_id}")
        
        return {"success": True, "message": "Application recorded successfully"}
    except Exception as e:
        logger.error(f"Error recording application: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Error recording application"
        )

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Job Recommendation API")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Job Recommendation API")
    db.close()

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    # Run the API
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=port, 
        reload=os.getenv("ENV", "development") == "development"
    )

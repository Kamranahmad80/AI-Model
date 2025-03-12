# File: api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from models.bert_embedding import get_bert_embedding
from models.recommendation_model import create_ranking_model
import pandas as pd

# Load FastAPI app
app = FastAPI(title="Job Recommendation API", description="AI-powered job recommendations")

# Define the request body schema
class RecommendationRequest(BaseModel):
    user_skills: str

# Load the trained model
MODEL_PATH = "models/job_recommender_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Load job data
jobs_df = pd.read_csv("dataset/jobs_cleaned.csv")

@app.get("/")
def home():
    return {"message": "Job Recommendation API is running!"}

@app.post("/recommend")
def recommend_jobs(request: RecommendationRequest):
    """
    Given user skills, return top recommended jobs.
    """
    # Extract user_skills from the request
    user_skills = request.user_skills
    
    # Convert user skills into embedding
    user_emb = get_bert_embedding(user_skills)

    # Compute similarity scores for each job
    recommendations = []
    for _, job in jobs_df.iterrows():
        job_emb = get_bert_embedding(job['cleaned_description'])
        combined_features = np.concatenate([user_emb, job_emb, np.abs(user_emb - job_emb)])
        score = model.predict(np.expand_dims(combined_features, axis=0))[0][0]
        recommendations.append((job["job_id"], job["job_title"], score))

    # Sort jobs by relevance score (higher is better)
    recommendations.sort(key=lambda x: x[2], reverse=True)

    return {
        "recommended_jobs": [
            {"job_id": job[0], "title": job[1], "score": float(job[2])}
            for job in recommendations[:5]  # Return top 5 jobs
        ]
    }

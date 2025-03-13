# File: api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pandas as pd
from models.bert_embedding import get_bert_embedding
from models.recommendation_model import create_ranking_model

app = FastAPI(title="Job Recommendation API", description="AI-powered job recommendations")

class RecommendationRequest(BaseModel):
    user_resume: str  # Resume text or URL

# Load the trained ranking model
MODEL_PATH = "models/job_recommender_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Load jobs data (ensure your CSV columns use lower-case keys)
jobs_df = pd.read_csv("dataset/jobs_cleaned.csv")
# For example, ensure your CSV has columns: "job_id", "job_title", "cleaned_description"

@app.get("/")
def home():
    return {"message": "Job Recommendation API is running!"}

@app.post("/recommend")
def recommend_jobs(request: RecommendationRequest):
    """
    Given a user's resume, compute BERT embedding and return top recommended jobs.
    """
    user_resume = request.user_resume
    user_emb = get_bert_embedding(user_resume)

    recommendations = []
    for _, job in jobs_df.iterrows():
        # We assume 'cleaned_description' is the lower-case column for job description
        job_emb = get_bert_embedding(job['cleaned_description'])
        combined_features = np.concatenate([user_emb, job_emb, np.abs(user_emb - job_emb)])
        score = model.predict(np.expand_dims(combined_features, axis=0))[0][0]
        recommendations.append((job["job_id"], job["job_title"], score))
    
    # Sort recommendations by score in descending order
    recommendations.sort(key=lambda x: x[2], reverse=True)

    return {
        "recommended_jobs": [
            {"job_id": job[0], "title": job[1], "score": float(job[2])}
            for job in recommendations[:5]
        ]
    }

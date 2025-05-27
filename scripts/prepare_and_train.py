"""
Prepare MongoDB job data and train a model for job recommendations.
This script will:
1. Extract job data from MongoDB
2. Process it into the required format
3. Train and save a model file
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import logging

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_connector import DatabaseConnector
from models.recommendation_model import create_ranking_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories if they don't exist
os.makedirs("dataset", exist_ok=True)
os.makedirs("models", exist_ok=True)

def extract_skills_from_job(job):
    """Extract all skills from a job document."""
    skills = []
    if isinstance(job.get('skills'), dict):
        if 'required' in job['skills'] and isinstance(job['skills']['required'], list):
            skills.extend(job['skills']['required'])
        if 'preferred' in job['skills'] and isinstance(job['skills']['preferred'], list):
            skills.extend(job['skills']['preferred'])
    elif isinstance(job.get('skills'), list):
        skills.extend(job['skills'])
    
    return list(set([s.lower() for s in skills if s]))

def get_job_description_text(job):
    """Extract and concatenate all descriptive text from a job."""
    texts = []
    
    # Add title
    if 'Title' in job and job['Title']:
        texts.append(str(job['Title']))
    
    # Add category
    if 'Category' in job and job['Category']:
        texts.append(str(job['Category']))
    
    # Add job description
    if isinstance(job.get('job_description'), dict):
        if 'overview' in job['job_description'] and job['job_description']['overview']:
            texts.append(str(job['job_description']['overview']))
        
        if 'responsibilities' in job['job_description'] and isinstance(job['job_description']['responsibilities'], list):
            for resp in job['job_description']['responsibilities']:
                texts.append(str(resp))
        
        if 'requirements' in job['job_description'] and isinstance(job['job_description']['requirements'], list):
            for req in job['job_description']['requirements']:
                texts.append(str(req))
    
    # Add tags
    if 'tags' in job and isinstance(job['tags'], list):
        texts.extend([str(tag) for tag in job['tags']])
    
    return " ".join(texts).lower()

def prepare_data():
    """Prepare job data for training."""
    logger.info("Starting data preparation...")
    
    # Connect to MongoDB
    db = DatabaseConnector()
    
    # Get all jobs
    all_jobs = db.get_all_jobs()
    logger.info(f"Retrieved {len(all_jobs)} jobs from database")
    
    if not all_jobs:
        logger.error("No jobs found in database. Cannot proceed with training.")
        return False
    
    # Process jobs into dataframe
    jobs_data = []
    for job in all_jobs:
        try:
            job_id = str(job.get('_id'))
            title = job.get('Title', '')
            company = job.get('Company', '')
            skills = extract_skills_from_job(job)
            description = get_job_description_text(job)
            
            jobs_data.append({
                'job_id': job_id,
                'title': title,
                'company': company,
                'skills': ','.join(skills),
                'cleaned_description': description
            })
        except Exception as e:
            logger.error(f"Error processing job: {e}")
    
    # Convert to dataframe
    jobs_df = pd.DataFrame(jobs_data)
    logger.info(f"Processed {len(jobs_df)} jobs successfully")
    
    # Save to CSV
    jobs_df.to_csv("dataset/jobs_cleaned.csv", index=False)
    logger.info("Saved jobs data to dataset/jobs_cleaned.csv")
    
    # Create dummy users based on skills
    all_skills = []
    for skills_str in jobs_df['skills']:
        all_skills.extend(skills_str.split(','))
    
    unique_skills = list(set(all_skills))
    logger.info(f"Found {len(unique_skills)} unique skills across all jobs")
    
    # Create some dummy users with different skill combinations
    num_dummy_users = min(50, len(unique_skills) * 2)
    users_data = []
    
    import random
    for i in range(num_dummy_users):
        # Select 3-7 random skills for each dummy user
        num_skills = random.randint(3, min(7, len(unique_skills)))
        user_skills = random.sample(unique_skills, num_skills)
        
        users_data.append({
            'user_id': f"dummy_user_{i}",
            'cleaned_skills': ' '.join(user_skills)
        })
    
    users_df = pd.DataFrame(users_data)
    users_df.to_csv("dataset/users_cleaned.csv", index=False)
    logger.info(f"Created {len(users_df)} dummy users for training")
    
    # Close DB connection
    db.close()
    return True

def train_simple_model():
    """Train a simple TF model for job recommendations."""
    logger.info("Starting model training...")
    
    # Load job data
    jobs_df = pd.read_csv("dataset/jobs_cleaned.csv")
    
    if len(jobs_df) == 0:
        logger.error("No job data available for training")
        return False
    
    # Create a simple model based on TF-IDF vectors for job descriptions
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Get all skills and descriptions
    all_descriptions = jobs_df['cleaned_description'].tolist()
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=100)
    job_vectors = vectorizer.fit_transform(all_descriptions)
    
    # Save the vectorizer
    import pickle
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    # Create a simple neural network model for recommendation
    input_dim = 100  # TF-IDF features
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Create synthetic training data
    # For each job, create a positive example (matching itself) and negative examples (non-matching)
    X = []
    y = []
    
    # Create positive examples - each job matches with itself
    for i in range(len(job_vectors.toarray())):
        X.append(job_vectors.toarray()[i])
        y.append(1.0)  # Positive match
    
    # Create negative examples - each job doesn't match with some other jobs
    import random
    for i in range(len(job_vectors.toarray())):
        # Choose 2 random non-matching jobs
        for _ in range(2):
            j = random.randint(0, len(job_vectors.toarray()) - 1)
            if i != j:  # Ensure it's a different job
                X.append(job_vectors.toarray()[j])
                y.append(0.0)  # Negative match
    
    X = np.array(X)
    y = np.array(y)
    
    # Train the model
    model.fit(X, y, epochs=20, batch_size=8, validation_split=0.2)
    
    # Save the model
    model.save("models/job_recommender_model.keras")
    logger.info("Model trained and saved to models/job_recommender_model.keras")
    
    # Save job vectors for later use
    np.save("models/job_vectors.npy", job_vectors.toarray())
    
    # Save job IDs
    with open("models/job_ids.json", "w") as f:
        json.dump(jobs_df['job_id'].tolist(), f)
    
    return True

def main():
    """Main execution function."""
    try:
        # Prepare the data
        if not prepare_data():
            logger.error("Data preparation failed. Exiting.")
            return
        
        # Train the model
        if not train_simple_model():
            logger.error("Model training failed. Exiting.")
            return
        
        logger.info("Successfully prepared data and trained model!")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()

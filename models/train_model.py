# File: models/train_model.py
import os
import numpy as np
import pandas as pd
from models.recommendation_model import create_ranking_model
from models.bert_embedding import get_bert_embedding

# Paths for preprocessed files (adjust if needed)
users_path = "./dataset/users_cleaned.csv"
jobs_path = "./dataset/jobs_cleaned.csv"

# Check if preprocessed files exist
if not os.path.exists(users_path) or not os.path.exists(jobs_path):
    raise FileNotFoundError("Preprocessed user or job file not found. Please run the preprocessing scripts first.")

# Load preprocessed user and job data
# Load preprocessed user and job data
users_df = pd.read_csv("./dataset/users_cleaned.csv")
jobs_df = pd.read_csv("./dataset/jobs_cleaned.csv")

# Ensure unique identifiers exist
if 'user_id' not in users_df.columns:
    users_df.reset_index(inplace=True)
    users_df.rename(columns={'index': 'user_id'}, inplace=True)
if 'job_id' not in jobs_df.columns:
    jobs_df.reset_index(inplace=True)
    jobs_df.rename(columns={'index': 'job_id'}, inplace=True)

# Generate dummy training data: pair each user with the first job in jobs_df
# Generate dummy training data: pair each user with the first job in jobs_df
dummy_interactions = []
for _, user in users_df.iterrows():
    job = jobs_df.iloc[0]  # pair each user with the first job
    # Ensure cleaned_skills is a string before splitting
    user_skills = str(user['cleaned_skills']).split()
    # Create a dummy label: 1 if any user skill appears in the job's description, else 0
    label = int(any(skill in job['cleaned_description'] for skill in user_skills))
    dummy_interactions.append({
        'user_id': user['user_id'],
        'job_id': job['job_id'],
        'label': label
    })

interactions_df = pd.DataFrame(dummy_interactions)

# Function to compute features for each user-job pair
def compute_features(row):
    # Get user embedding
    user_text = users_df.loc[users_df['user_id'] == row['user_id'], 'cleaned_skills'].values[0]
    user_emb = get_bert_embedding(user_text)
    # Get job embedding
    job_text = jobs_df.loc[jobs_df['job_id'] == row['job_id'], 'cleaned_description'].values[0]
    job_emb = get_bert_embedding(job_text)
    # Combine embeddings: concatenate user_emb, job_emb, and their absolute difference
    combined = np.concatenate([user_emb, job_emb, np.abs(user_emb - job_emb)])
    return combined

# Create training features and labels
features = []
labels = []
for _, row in interactions_df.iterrows():
    try:
        feat = compute_features(row)
        features.append(feat)
        labels.append(row['label'])
    except Exception as e:
        print(f"Error processing row: {e}")

X = np.array(features)
y = np.array(labels)

print("Training data shape:", X.shape)

# Define input dimension for the ranking model (e.g., if using BERT's 768 dimensions, then 768*3)
input_dim = X.shape[1]
model = create_ranking_model(input_dim)

# Train the model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Save the model for later use
model.save("models/job_recommender_model.keras")
print("Model trained and saved.")

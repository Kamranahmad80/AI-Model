"""
Advanced Job Recommendation Model
This file implements a complete AI-based job recommendation system.
It handles:
1. Data processing and feature extraction
2. Model creation and training
3. Prediction and recommendation generation
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Make TensorFlow less verbose
tf.get_logger().setLevel('ERROR')

# Import database connector
from database.db_connector import DatabaseConnector

class JobRecommender:
    """Complete job recommendation system"""
    
    def __init__(self):
        """Initialize the recommendation system"""
        self.db = DatabaseConnector()
        self.model = None
        self.tfidf_vectorizer = None
        self.feature_scaler = None
        self.skill_list = []
        self.model_path = Path('models/job_recommender_model.keras')
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Try to load existing models and data
        self._load_existing_models()
    
    def _load_existing_models(self):
        """Try to load existing models and data"""
        try:
            # Check if the model exists
            if self.model_path.exists():
                logger.info(f"Loading existing model from {self.model_path}")
                self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
                self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # Load TF-IDF vectorizer
            tfidf_path = Path('models/tfidf_vectorizer.pkl')
            if tfidf_path.exists():
                with open(tfidf_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                    logger.info("Loaded TF-IDF vectorizer")
            
            # Load feature scaler
            scaler_path = Path('models/feature_scaler.pkl')
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                    logger.info("Loaded feature scaler")
            
            # Load skill list
            skills_path = Path('models/unique_skills.json')
            if skills_path.exists():
                with open(skills_path, 'r') as f:
                    self.skill_list = json.load(f)
                    logger.info(f"Loaded {len(self.skill_list)} skills")
        
        except Exception as e:
            logger.error(f"Error loading existing models: {e}")
            self.model = None
            self.tfidf_vectorizer = None
            self.feature_scaler = None
            self.skill_list = []
    
    def _extract_skills_from_job(self, job):
        """Extract skills from job document"""
        skills = []
        if isinstance(job.get('skills'), dict):
            # Get required skills
            if 'required' in job['skills'] and isinstance(job['skills']['required'], list):
                skills.extend(job['skills']['required'])
            # Get preferred skills
            if 'preferred' in job['skills'] and isinstance(job['skills']['preferred'], list):
                skills.extend(job['skills']['preferred'])
        elif isinstance(job.get('skills'), list):
            skills.extend(job['skills'])
        
        # Also extract from tags
        if isinstance(job.get('tags'), list):
            skills.extend(job['tags'])
        
        # Clean and normalize skills
        return list(set([s.lower() for s in skills if s]))
    
    def _get_job_description_text(self, job):
        """Extract descriptive text from job"""
        texts = []
        
        # Add title and company
        if 'Title' in job and job['Title']:
            texts.append(str(job['Title']))
        if 'Company' in job and job['Company']:
            texts.append(str(job['Company']))
        if 'Category' in job and job['Category']:
            texts.append(str(job['Category']))
        
        # Add job description
        if isinstance(job.get('job_description'), dict):
            if 'overview' in job['job_description'] and job['job_description']['overview']:
                texts.append(str(job['job_description']['overview']))
            
            # Add responsibilities
            if 'responsibilities' in job['job_description'] and isinstance(job['job_description']['responsibilities'], list):
                for resp in job['job_description']['responsibilities']:
                    texts.append(str(resp))
            
            # Add requirements
            if 'requirements' in job['job_description'] and isinstance(job['job_description']['requirements'], list):
                for req in job['job_description']['requirements']:
                    texts.append(str(req))
        
        # Add tags
        if 'tags' in job and isinstance(job['tags'], list):
            texts.extend([str(tag) for tag in job['tags']])
        
        return " ".join(texts).lower()
    
    def _create_feature_extractors(self, jobs):
        """Create or update feature extractors for job data"""
        logger.info("Creating feature extractors")
        
        # Collect all job descriptions
        descriptions = []
        all_skills = set()
        salary_values = []
        experience_values = []
        
        for job in jobs:
            # Get description
            description = self._get_job_description_text(job)
            descriptions.append(description)
            
            # Get skills
            skills = self._extract_skills_from_job(job)
            all_skills.update(skills)
            
            # Get salary
            salary = 0
            if 'salary' in job:
                if isinstance(job['salary'], int):
                    salary = job['salary']
                elif isinstance(job['salary'], dict) and 'min' in job['salary']:
                    salary = job['salary']['min']
            salary_values.append(salary)
            
            # Get experience
            experience = 0
            if 'experienceYears' in job and isinstance(job['experienceYears'], dict):
                min_exp = job['experienceYears'].get('min', 0)
                max_exp = job['experienceYears'].get('max', 0)
                experience = (min_exp + max_exp) / 2 if max_exp > 0 else min_exp
            experience_values.append(experience)
        
        # Create TF-IDF vectorizer for job descriptions
        if not self.tfidf_vectorizer:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            self.tfidf_vectorizer.fit(descriptions)
            
            # Save the vectorizer
            with open('models/tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
        
        # Save the skill list
        self.skill_list = list(all_skills)
        with open('models/unique_skills.json', 'w') as f:
            json.dump(self.skill_list, f)
        
        # Create feature scaler for numeric features
        numeric_features = np.array([salary_values, experience_values]).T
        if not self.feature_scaler:
            self.feature_scaler = StandardScaler()
            self.feature_scaler.fit(numeric_features)
            
            # Save the scaler
            with open('models/feature_scaler.pkl', 'wb') as f:
                pickle.dump(self.feature_scaler, f)
        
        logger.info(f"Feature extractors created: {len(descriptions)} descriptions, {len(self.skill_list)} skills")
    
    def _extract_job_features(self, job):
        """Extract features from a job for model input"""
        # Ensure we have feature extractors
        if not self.tfidf_vectorizer or not self.feature_scaler or not self.skill_list:
            logger.error("Feature extractors not initialized")
            return None
        
        try:
            # Get job description features
            description = self._get_job_description_text(job)
            desc_vector = self.tfidf_vectorizer.transform([description]).toarray()[0]
            
            # Get skills features
            job_skills = self._extract_skills_from_job(job)
            skill_vector = np.zeros(len(self.skill_list))
            
            for skill in job_skills:
                if skill in self.skill_list:
                    idx = self.skill_list.index(skill)
                    skill_vector[idx] = 1
            
            # Get salary feature
            salary = 0
            if 'salary' in job:
                if isinstance(job['salary'], int):
                    salary = job['salary']
                elif isinstance(job['salary'], dict) and 'min' in job['salary']:
                    salary = job['salary']['min']
            
            # Get experience feature
            experience = 0
            if 'experienceYears' in job and isinstance(job['experienceYears'], dict):
                min_exp = job['experienceYears'].get('min', 0)
                max_exp = job['experienceYears'].get('max', 0)
                experience = (min_exp + max_exp) / 2 if max_exp > 0 else min_exp
            
            # Scale numeric features
            numeric_features = np.array([[salary, experience]])
            scaled_numeric = self.feature_scaler.transform(numeric_features)[0]
            
            # Combine all features
            features = np.concatenate([
                desc_vector, 
                skill_vector,
                scaled_numeric
            ])
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting job features: {e}")
            return None
    
    def _extract_resume_features(self, resume_data, user_profile=None):
        """Extract features from a resume for model input"""
        # Ensure we have feature extractors
        if not self.tfidf_vectorizer or not self.feature_scaler or not self.skill_list:
            logger.error("Feature extractors not initialized")
            return None
        
        try:
            logger.info(f"Extracting features from resume data: {type(resume_data)}")
            
            # Validate input data
            if resume_data is None:
                logger.warning("Resume data is None, using empty data")
                resume_data = {
                    'skills': ["python", "javascript", "web development"],
                    'experience': [],
                    'education': []
                }
            
            # Handle case where skills might be missing or empty
            resume_skills = []
            if 'skills' in resume_data:
                if isinstance(resume_data['skills'], list):
                    resume_skills = resume_data['skills']
                elif isinstance(resume_data['skills'], str):
                    resume_skills = [resume_data['skills']]
                    
            # If still no skills, add default skills
            if not resume_skills:
                resume_skills = ["python", "javascript", "web development"]
                logger.warning("No skills found in resume, using default skills")
            
            logger.info(f"Extracted {len(resume_skills)} skills from resume")
            
            # Create skill vector
            skill_vector = np.zeros(len(self.skill_list))
            matched_skills = 0
            for skill in resume_skills:
                if not skill:
                    continue
                    
                skill_lower = skill.lower()
                if skill_lower in self.skill_list:
                    idx = self.skill_list.index(skill_lower)
                    skill_vector[idx] = 1
                    matched_skills += 1
                    
            logger.info(f"Matched {matched_skills} skills with our skill database")
            
            # Create description vector from skills and experience
            resume_text = " ".join(resume_skills)
            if 'experience' in resume_data and isinstance(resume_data['experience'], list):
                for exp in resume_data['experience']:
                    if isinstance(exp, dict) and 'description' in exp:
                        resume_text += " " + exp['description']
                    if isinstance(exp, dict) and 'role' in exp:
                        resume_text += " " + exp['role']
            
            desc_vector = self.tfidf_vectorizer.transform([resume_text]).toarray()[0]
            
            # Get preferred salary and experience from user profile
            preferred_salary = 0
            experience_years = 0
            
            if user_profile:
                preferred_salary = user_profile.get('preferred_salary', 0)
                
                # Calculate total experience years from resume
                if 'experience' in resume_data and isinstance(resume_data['experience'], list):
                    for exp in resume_data['experience']:
                        if isinstance(exp, dict):
                            start_year = 0
                            end_year = 0
                            
                            if 'start_year' in exp:
                                try:
                                    start_year = int(exp['start_year'])
                                except:
                                    start_year = 0
                            
                            if 'end_year' in exp:
                                if exp['end_year'].lower() == 'present':
                                    from datetime import datetime
                                    end_year = datetime.now().year
                                else:
                                    try:
                                        end_year = int(exp['end_year'])
                                    except:
                                        end_year = 0
                            
                            if start_year > 0 and end_year > 0:
                                experience_years += (end_year - start_year)
            
            # Scale numeric features
            numeric_features = np.array([[preferred_salary, experience_years]])
            scaled_numeric = self.feature_scaler.transform(numeric_features)[0]
            
            # Combine all features
            features = np.concatenate([
                desc_vector, 
                skill_vector,
                scaled_numeric
            ])
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting resume features: {e}")
            return None
    
    def _build_model(self, input_dim):
        """Build the recommendation model"""
        logger.info(f"Building model with input dimension {input_dim}")
        
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
        
        return model
    
    def _create_training_data(self, jobs):
        """Create training data from job listings"""
        logger.info("Creating training data")
        
        # Extract features for each job
        job_features = []
        job_ids = []
        
        for job in jobs:
            features = self._extract_job_features(job)
            if features is not None:
                job_features.append(features)
                job_ids.append(str(job.get('_id', '')))
        
        if not job_features:
            logger.error("No job features could be extracted")
            return None, None, None
        
        X = np.array(job_features)
        
        # Create synthetic training data - match jobs with similar skills
        train_X = []
        train_y = []
        
        # Self-similarity (positive examples)
        for i in range(len(X)):
            train_X.append(X[i])
            train_y.append(1.0)  # Perfect match
        
        # Generate negative examples (non-matches)
        import random
        for i in range(len(X)):
            for _ in range(2):  # Create 2 negative examples for each job
                j = random.randint(0, len(X) - 1)
                if i != j:  # Ensure different jobs
                    # Mix features for negative examples
                    mixed_features = X[i] * 0.2 + X[j] * 0.8
                    train_X.append(mixed_features)
                    train_y.append(0.0)  # Non-match
        
        return np.array(train_X), np.array(train_y), job_ids
    
    def create_and_train_model(self):
        """Create and train the recommendation model"""
        try:
            logger.info("Starting model creation and training")
            
            # Get all jobs from the database
            all_jobs = self.db.get_all_jobs()
            if not all_jobs or len(all_jobs) == 0:
                logger.error("No jobs found in the database")
                return False
            
            logger.info(f"Retrieved {len(all_jobs)} jobs from database")
            
            # Create feature extractors
            self._create_feature_extractors(all_jobs)
            
            # Create training data
            X, y, job_ids = self._create_training_data(all_jobs)
            if X is None or y is None:
                logger.error("Failed to create training data")
                return False
            
            logger.info(f"Created training dataset with {len(X)} samples")
            
            # Save job IDs for reference
            with open('models/job_ids.json', 'w') as f:
                json.dump(job_ids, f)
            
            # Build and train the model
            input_dim = X.shape[1]
            self.model = self._build_model(input_dim)
            
            # Train the model
            self.model.fit(X, y, epochs=20, batch_size=16, validation_split=0.2)
            
            # Save the model
            self.model.save(str(self.model_path))
            logger.info(f"Model saved to {self.model_path}")
            
            # Save model metadata
            with open('models/model_metadata.json', 'w') as f:
                json.dump({
                    'input_dim': input_dim,
                    'feature_count': {
                        'tfidf': len(self.tfidf_vectorizer.get_feature_names_out()),
                        'skills': len(self.skill_list),
                        'numeric': 2  # salary and experience
                    },
                    'job_count': len(all_jobs),
                    'training_samples': len(X)
                }, f)
            
            logger.info("Model creation and training completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error in model creation and training: {e}")
            return False
    
    def _get_simple_recommendations(self, resume_data, all_jobs, num_recommendations=5):
        """Get simple recommendations based on skill matching only"""
        try:
            logger.info("Using simple skill-based matching for recommendations")
            
            # Extract skills from resume
            resume_skills = []
            if resume_data and 'skills' in resume_data:
                if isinstance(resume_data['skills'], list):
                    resume_skills = resume_data['skills']
                elif isinstance(resume_data['skills'], str):
                    resume_skills = [resume_data['skills']]
            
            # Add default skills if none found
            if not resume_skills:
                resume_skills = ["python", "javascript", "web development", "software engineering"]
                logger.warning("Using default skills for simple matching")
            
            # Convert to lowercase
            resume_skills = [skill.lower() for skill in resume_skills if skill]
            logger.info(f"Using {len(resume_skills)} skills for simple matching")
            
            # Score each job based on skill matches
            scored_jobs = []
            for job in all_jobs:
                # Extract job skills
                job_skills = []
                
                # Check if skills is a dictionary with required/preferred
                if isinstance(job.get('skills'), dict):
                    if 'required' in job['skills'] and isinstance(job['skills']['required'], list):
                        job_skills.extend([s.lower() for s in job['skills']['required'] if s])
                    if 'preferred' in job['skills'] and isinstance(job['skills']['preferred'], list):
                        job_skills.extend([s.lower() for s in job['skills']['preferred'] if s])
                
                # Check if skills is a list
                elif isinstance(job.get('skills'), list):
                    job_skills = [s.lower() for s in job['skills'] if s]
                
                # Also check tags for additional skills
                if isinstance(job.get('tags'), list):
                    job_skills.extend([tag.lower() for tag in job['tags'] if tag])
                
                # Calculate matching skills
                matching_skills = set(resume_skills).intersection(set(job_skills))
                
                # Calculate score - base score is always at least 0.2 to ensure all jobs get considered
                base_score = 0.2
                skill_score = len(matching_skills) / max(1, len(job_skills)) if job_skills else 0
                match_score = base_score + (0.8 * skill_score)  # Scale skill score to 0.2-1.0 range
                
                # Create job copy with match information
                job_copy = dict(job)
                job_copy['match_score'] = match_score
                job_copy['matching_skills'] = list(matching_skills)
                job_copy['match_factors'] = {
                    'skill_match': skill_score,
                    'base_score': base_score
                }
                
                # Add reasoning for the match
                if matching_skills:
                    job_copy['match_reason'] = f"Matched {len(matching_skills)} skills: {', '.join(list(matching_skills)[:3])}"
                    if len(matching_skills) > 3:
                        job_copy['match_reason'] += f" and {len(matching_skills) - 3} more"
                else:
                    job_copy['match_reason'] = "Potential match based on job category"
                
                scored_jobs.append(job_copy)
            
            # Sort by match score
            scored_jobs.sort(key=lambda x: x['match_score'], reverse=True)
            
            # Return top N recommendations
            return scored_jobs[:num_recommendations]
        
        except Exception as e:
            logger.error(f"Error in simple recommendations: {e}")
            return []
    
    def get_recommendations(self, resume_data, user_profile=None, num_recommendations=5):
        """Get job recommendations based on resume and user profile"""
        try:
            # Check if model exists
            if self.model is None:
                logger.warning("Model not loaded or trained, will use similarity matching")
            
            # Get all jobs
            all_jobs = self.db.get_all_jobs()
            if not all_jobs:
                logger.warning("No jobs available in the database")
                return []
            
            logger.info(f"Generating recommendations from {len(all_jobs)} available jobs")
            
            # Extract resume features
            resume_features = self._extract_resume_features(resume_data, user_profile)
            if resume_features is None:
                logger.warning("Could not extract resume features, using simplified matching")
                # Create simple recommendations based on skill matching only
                return self._get_simple_recommendations(resume_data, all_jobs, num_recommendations)
            
            # Calculate match scores for all jobs
            job_scores = []
            
            for job in all_jobs:
                # Extract job features
                job_features = self._extract_job_features(job)
                if job_features is None:
                    continue
                
                # Calculate similarity using dot product
                similarity = np.dot(resume_features, job_features) / (np.linalg.norm(resume_features) * np.linalg.norm(job_features))
                
                # Use model for final prediction
                combined_features = np.concatenate([resume_features, job_features, np.abs(resume_features - job_features)])
                combined_features = np.expand_dims(combined_features, axis=0)
                
                prediction = float(self.model.predict(combined_features)[0][0])
                
                # Calculate skill match score for explanation
                user_skills = resume_data.get('skills', [])
                if isinstance(user_skills, str):
                    user_skills = [user_skills]
                
                required_skills = []
                if 'skills' in job and isinstance(job['skills'], dict) and 'required' in job['skills']:
                    required_skills = job['skills']['required']
                
                matching_skills = set([s.lower() for s in user_skills]).intersection([s.lower() for s in required_skills])
                skill_match = len(matching_skills) / max(1, len(required_skills)) if required_skills else 0
                
                # Create job copy with match information
                job_copy = dict(job)
                job_copy['match_score'] = prediction
                job_copy['similarity_score'] = float(similarity)
                job_copy['matching_skills'] = list(matching_skills)
                job_copy['match_factors'] = {
                    'model_score': prediction,
                    'similarity_score': float(similarity),
                    'skill_match': skill_match
                }
                
                # Add reasoning for the match
                if prediction >= 0.7:
                    job_copy['match_reason'] = f"Strong match based on {len(matching_skills)} matching skills and your experience"
                elif prediction >= 0.4:
                    job_copy['match_reason'] = "Good match based on your profile and requirements"
                else:
                    job_copy['match_reason'] = "Potential match worth exploring"
                
                job_scores.append(job_copy)
            
            # Sort by match score
            job_scores.sort(key=lambda x: x['match_score'], reverse=True)
            
            # Return top N recommendations
            return job_scores[:num_recommendations]
        
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []


# Main execution when run directly
if __name__ == "__main__":
    try:
        # Create the recommender
        recommender = JobRecommender()
        
        # Create and train the model
        success = recommender.create_and_train_model()
        
        if success:
            logger.info("✅ Model created and trained successfully")
        else:
            logger.error("❌ Failed to create model")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")

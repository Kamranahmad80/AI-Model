import tensorflow as tf
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from database.db_connector import DatabaseConnector
from services.job_matcher_helper import calculate_skill_match_score, calculate_text_similarity
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class JobMatcher:
    def __init__(self):
        self.db = DatabaseConnector()
        self.mlb = MultiLabelBinarizer()
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        # Load or initialize skill encoder
        self.initialize_skill_encoder()
        logging.info("JobMatcher initialized")
        
    def initialize_skill_encoder(self):
        """Initialize the skill encoder with all possible skills"""
        # Get all unique skills from the database
        all_skills = self.db.get_all_skills()
        logging.info(f"Loaded {len(all_skills)} unique skills for encoding")
        self.mlb.fit([all_skills])
        
        # Initialize TF-IDF model if we have job descriptions
        self._initialize_tfidf_model()
    
    def encode_skills(self, skills: List[str]) -> np.ndarray:
        """Convert skills list to binary vector"""
        # Ensure skills are lowercased and unique
        normalized_skills = list(set([s.lower() for s in skills if s]))
        return self.mlb.transform([normalized_skills])[0]
    
    def encode_experience(self, experience: List[Dict]) -> Dict[str, Any]:
        """Calculate experience metrics including years and recency"""
        from datetime import datetime
        
        total_years = 0
        current_year = datetime.now().year
        most_recent_year = 0
        roles = []
        companies = []
        
        for exp in experience:
            # Extract years
            start_year = int(exp['start_year'])
            end_year = current_year if exp['end_year'].lower() == 'present' else int(exp['end_year'])
            duration = end_year - start_year
            total_years += duration
            
            # Track most recent experience
            most_recent_year = max(most_recent_year, end_year if end_year != current_year else current_year)
            
            # Get roles and companies
            if 'role' in exp and exp['role']:
                roles.append(exp['role'])
            if 'company' in exp and exp['company']:
                companies.append(exp['company'])
        
        # Experience recency score (higher is more recent)
        recency_score = 1.0 if most_recent_year == current_year else max(0.0, 1.0 - (current_year - most_recent_year) / 10.0)
        
        return {
            'total_years': total_years,
            'recency_score': recency_score,
            'roles': roles,
            'companies': companies
        }
    
    def _initialize_tfidf_model(self):
        """Initialize the TF-IDF model with job descriptions"""
        try:
            # Get all job descriptions from database
            all_job_descriptions = self.db.get_all_job_descriptions()
            if all_job_descriptions and len(all_job_descriptions) > 0:
                # Fit the TF-IDF vectorizer on all job descriptions
                self.tfidf_vectorizer.fit(all_job_descriptions)
                logging.info(f"TF-IDF model initialized with {len(all_job_descriptions)} job descriptions")
            else:
                logging.warning("No job descriptions available for TF-IDF initialization")
        except Exception as e:
            logging.error(f"Error initializing TF-IDF model: {str(e)}")
    
    def _calculate_skill_match_score(self, job_skills: List[str], user_skills: List[str]) -> float:
        """Calculate the skill match score between job requirements and user skills"""
        if not job_skills or not user_skills:
            return 0.0
        
        # Count the number of matching skills
        job_skills_lower = [s.lower() for s in job_skills]
        user_skills_lower = [s.lower() for s in user_skills]
        
        matching_skills = set(job_skills_lower).intersection(user_skills_lower)
        
        # Calculate match score as a ratio of matching skills to required skills
        if len(job_skills_lower) > 0:
            base_score = len(matching_skills) / len(job_skills_lower)
            
            # Bonus for having more than the required skills
            bonus = min(0.2, 0.02 * (len(matching_skills) - 1)) if len(matching_skills) > 1 else 0
            
            return min(1.0, base_score + bonus)
        return 0.0

    def _calculate_text_similarity(self, job_description: str, user_profile_text: str) -> float:
        """Calculate text similarity between job description and user profile"""
        if not job_description or not user_profile_text:
            return 0.0
            
        try:
            # Transform texts to TF-IDF vectors
            job_vector = self.tfidf_vectorizer.transform([job_description])
            user_vector = self.tfidf_vectorizer.transform([user_profile_text])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(job_vector, user_vector)[0][0]
            return float(similarity)
        except Exception as e:
            logging.error(f"Error calculating text similarity: {str(e)}")
            return 0.0
    
    def _calculate_location_match(self, job_location: Any, user_location: Any) -> float:
        """Calculate location match score based on location codes or text"""
        if job_location is None or user_location is None:
            return 0.5  # Neutral score for missing location data
            
        # If locations are codes (integers)
        if isinstance(job_location, int) and isinstance(user_location, int):
            if job_location == user_location:
                return 1.0  # Exact match
            elif job_location // 100 == user_location // 100:  # Same region
                return 0.8
            elif job_location // 10000 == user_location // 10000:  # Same country
                return 0.5
            return 0.2  # Different countries
        
        # If locations are strings, check for substring match
        if isinstance(job_location, str) and isinstance(user_location, str):
            job_loc_lower = job_location.lower()
            user_loc_lower = user_location.lower()
            
            if job_loc_lower == user_loc_lower:
                return 1.0  # Exact match
            elif job_loc_lower in user_loc_lower or user_loc_lower in job_loc_lower:
                return 0.8  # Partial match
            
            # Split into parts and check for any matches
            job_parts = set(job_loc_lower.split())
            user_parts = set(user_loc_lower.split())
            common_parts = job_parts.intersection(user_parts)
            
            if common_parts:
                return 0.5 + (0.5 * len(common_parts) / max(len(job_parts), len(user_parts)))
                
        return 0.3  # Different locations

    def get_recommendations(self, model: tf.keras.Model = None, resume_data: dict = None, 
                          user_profile: dict = None, num_recommendations: int = 5) -> List[Dict]:
        """Get job recommendations based on resume and user profile"""        
        # Prepare available jobs
        all_jobs = self.db.get_all_jobs()
        if not all_jobs:
            logging.warning("No jobs available in the database")
            return []
            
        logging.info(f"Generating recommendations from {len(all_jobs)} available jobs")
        
        # If AI model is available, use it for predictions
        if model is not None and resume_data and user_profile:
            return self._get_model_recommendations(model, resume_data, user_profile, all_jobs, num_recommendations)
        else:
            # Fallback to rule-based matching if model is not available
            return self._get_rule_based_recommendations(resume_data, user_profile, all_jobs, num_recommendations)
    
    def _get_model_recommendations(self, model: tf.keras.Model, resume_data: dict, 
                               user_profile: dict, all_jobs: List[Dict], num_recommendations: int) -> List[Dict]:
        """Get recommendations using the trained AI model"""
        try:
            # Encode input features
            skills_vector = self.encode_skills(resume_data.get('skills', []))
            experience_data = self.encode_experience(resume_data.get('experience', []))
            
            # Combine features for model input
            input_features = np.concatenate([
                skills_vector,
                np.array([experience_data['total_years']]),
                np.array([experience_data['recency_score']]),
                np.array([user_profile.get('preferred_salary', 0)]),
                np.array([user_profile.get('preferred_location_code', 0)])
            ])
            
            # Get model predictions
            predictions = model.predict(np.expand_dims(input_features, axis=0))[0]
            
            # Get top N job categories (or use all jobs if categories aren't properly set)
            top_indices = np.argsort(predictions)[-num_recommendations:][::-1]
            
            # Start with all jobs first
            recommendations = []
            
            # Try to use categories if they exist in the job data
            category_jobs_found = False
            for idx in top_indices:
                jobs = self.db.get_jobs_by_category(idx)
                if jobs:
                    category_jobs_found = True
                    for job in jobs:
                        # Ensure the job has an id that's a string (ObjectId converted to string)
                        job_id = job.get('id', str(job.get('_id', '')))
                        
                        job_copy = job.copy()
                        job_copy['id'] = job_id
                        job_copy['match_score'] = float(predictions[idx]) if idx < len(predictions) else 0.5
                        job_copy['match_factors'] = {
                            'model_score': job_copy['match_score'],
                            'skill_match': self._calculate_skill_match_score(
                                job.get('required_skills', []) or job.get('skills', {}).get('required', []), 
                                resume_data.get('skills', [])
                            )
                        }
                        job_copy['match_reason'] = 'AI model recommendation based on profile similarity'
                        recommendations.append(job_copy)
            
            # If no category-based jobs found, fallback to scoring all jobs
            if not category_jobs_found:
                logging.info("No category-based jobs found, scoring all available jobs")
                recommendations = self._score_all_jobs(all_jobs, resume_data, predictions)
            
            # Sort by match score
            recommendations.sort(key=lambda x: x.get('match_score', 0), reverse=True)
            
            return recommendations[:num_recommendations]
        except Exception as e:
            logging.error(f"Error in model recommendations: {str(e)}")
            # Return an empty list if there's an error
            return []
            
    def _score_all_jobs(self, all_jobs: List[Dict], resume_data: dict, predictions: np.ndarray) -> List[Dict]:
        """Score all available jobs when categories are not available"""
        try:
            scored_jobs = []
            
            for job in all_jobs:
                # Basic validation
                if not job:
                    continue
                    
                # Ensure the job has an id that's a string (ObjectId converted to string)
                job_id = job.get('id', str(job.get('_id', '')))
                
                # Calculate basic skill match
                user_skills = resume_data.get('skills', [])
                required_skills = job.get('required_skills', []) or job.get('skills', {}).get('required', [])
                skill_match = self._calculate_skill_match_score(required_skills, user_skills)
                
                # Calculate a score based on skill match since we can't use model prediction directly
                score = skill_match * 0.8 + 0.2  # Base score so nothing is zero
                
                job_copy = job.copy()
                job_copy['id'] = job_id
                job_copy['match_score'] = float(score)
                job_copy['match_factors'] = {
                    'skill_match': skill_match
                }
                job_copy['match_reason'] = 'Matched based on skills in your profile'
                
                scored_jobs.append(job_copy)
                
            return scored_jobs
        except Exception as e:
            logging.error(f"Error scoring all jobs: {str(e)}")
            return []  # Return empty list on error
    
    def _get_rule_based_recommendations(self, resume_data: dict, user_profile: dict, 
                                     all_jobs: List[Dict], num_recommendations: int) -> List[Dict]:
        """Get recommendations using rule-based matching when model is unavailable"""
        try:
            # If no resume data, return highest quality jobs
            if not resume_data or not resume_data.get('skills'):
                logging.warning("No resume data for matching, returning generic recommendations")
                # Sort by company reputation or date (newest first)
                return sorted(all_jobs, key=lambda j: j.get('date_posted', '2000-01-01'), reverse=True)[:num_recommendations]
            
            # Calculate match scores
            scored_jobs = []
            
            # Safely extract data with defaults
            user_skills = []
            if resume_data:
                if isinstance(resume_data, dict):
                    user_skills = resume_data.get('skills', [])
                    if isinstance(user_skills, str):
                        user_skills = [user_skills]
                    experience_list = resume_data.get('experience', [])
                else:
                    # Handle case where resume_data might be a pydantic model
                    try:
                        user_skills = getattr(resume_data, 'skills', [])
                        experience_list = getattr(resume_data, 'experience', [])
                    except Exception as e:
                        logging.error(f"Error accessing resume data: {e}")
                        experience_list = []
            else:
                experience_list = []
                
            # Encode experience data
            try:
                experience_data = self.encode_experience(experience_list)
            except Exception as e:
                logging.error(f"Error encoding experience: {e}")
                experience_data = {'roles': [], 'total_years': 0, 'recency_score': 0}
                
            # Create a combined text of skills and experience
            user_profile_text = ' '.join(user_skills) if isinstance(user_skills, list) else ''
            if experience_data and experience_data.get('roles'):
                user_profile_text += ' ' + ' '.join(experience_data['roles'])
            
            # Extract user preferences safely
            user_preferred_location = None
            user_preferred_salary = 0
            if user_profile:
                try:
                    user_preferred_location = user_profile.get('preferred_location_code')
                    user_preferred_salary = user_profile.get('preferred_salary', 0)
                except Exception as e:
                    logging.error(f"Error accessing user profile data: {e}")
            
            for job in all_jobs:
                # Calculate different match factors
                try:
                    # Use the helper function to safely calculate skill match
                    skill_match = calculate_skill_match_score(job.get('required_skills', []), user_skills)
                except Exception as e:
                    logging.error(f"Error calculating skill match: {e}")
                    skill_match = 0.3  # Default to a moderate match on error
                
                # Use the helper function for text similarity
                text_match = calculate_text_similarity(job.get('description', ''), user_profile_text)
                location_match = self._calculate_location_match(job.get('location_code'), user_preferred_location)
                
                # Calculate salary match
                salary_min = job.get('salary_range', {}).get('min', 0)
                salary_max = job.get('salary_range', {}).get('max', 0)
                
                salary_match = 0.5  # Neutral by default
                if salary_min and salary_max and user_preferred_salary:
                    if salary_min <= user_preferred_salary <= salary_max:
                        salary_match = 1.0  # Preferred salary in range
                    elif user_preferred_salary < salary_min:
                        # Salary higher than preferred (good, but not perfect)
                        ratio = min(1.0, salary_min / max(1, user_preferred_salary))
                        salary_match = 0.5 + (0.5 * (1.0 - ratio))
                    else:  # user_preferred_salary > salary_max
                        # Salary lower than preferred
                        ratio = min(1.0, salary_max / max(1, user_preferred_salary))
                        salary_match = ratio
                
                # Weight the different factors
                match_score = (
                    skill_match * 0.5 +      # Skills are most important
                    text_match * 0.2 +       # Text similarity adds value
                    location_match * 0.2 +   # Location is important 
                    salary_match * 0.1       # Salary is least weighted
                )
                
                job_copy = job.copy()  # Don't modify the original job
                job_copy['match_score'] = match_score
                job_copy['match_factors'] = {
                    'skill_match': skill_match,
                    'text_match': text_match,
                    'location_match': location_match,
                    'salary_match': salary_match
                }
                
                scored_jobs.append(job_copy)
            
            # Sort by match score (descending)
            scored_jobs.sort(key=lambda x: x['match_score'], reverse=True)
            
            return scored_jobs[:num_recommendations]
            
        except Exception as e:
            logging.error(f"Error in rule-based matching: {str(e)}")
            # Last resort: return random jobs
            import random
            random_jobs = all_jobs.copy()
            random.shuffle(random_jobs)
            return random_jobs[:num_recommendations]
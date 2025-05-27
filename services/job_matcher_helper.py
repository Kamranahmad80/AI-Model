"""
Helper functions for the job matcher service to handle edge cases
"""
import logging

def calculate_skill_match_score(required_skills, user_skills):
    """Calculate the match score between required job skills and user skills"""
    try:
        # Handle cases where input is not a list
        if not isinstance(required_skills, list):
            if isinstance(required_skills, str):
                required_skills = [required_skills]
            else:
                required_skills = []
                
        if not isinstance(user_skills, list):
            if isinstance(user_skills, str):
                user_skills = [user_skills]
            else:
                user_skills = []
                
        # If no skills to match, return a default score
        if not required_skills:
            return 0.65  # Higher neutral score when no skills required
            
        # Normalize skills (lowercase)
        req_skills_norm = [str(s).lower() for s in required_skills if s]
        user_skills_norm = [str(s).lower() for s in user_skills if s]
        
        # If user has no skills, return a moderate match
        if not user_skills_norm:
            return 0.4  # Moderate score when user has no skills
            
        # Find matching skills
        matching_skills = set(req_skills_norm).intersection(set(user_skills_norm))
        
        # Calculate match percentage
        if len(req_skills_norm) > 0:
            # Base score that ensures even zero matches get some score
            base_score = 0.35
            # Calculate match percentage with scaling to ensure a reasonable minimum
            match_score = base_score + (0.65 * len(matching_skills) / len(req_skills_norm))
            return min(1.0, match_score)  # Cap at 1.0
        else:
            return 0.65  # Higher default score if no required skills
    except Exception as e:
        logging.error(f"Error in skill matching: {e}")
        return 0.45  # Default moderate score on error

def calculate_text_similarity(job_text, user_text):
    """Calculate text similarity between job description and user profile text"""
    try:
        if not job_text or not user_text:
            return 0.3  # Default score when missing data
            
        # Use simple word overlap for similarity
        job_words = set(str(job_text).lower().split())
        user_words = set(str(user_text).lower().split())
        
        # Calculate Jaccard similarity
        if not job_words or not user_words:
            return 0.3
            
        intersection = job_words.intersection(user_words)
        union = job_words.union(user_words)
        
        return len(intersection) / len(union)
    except Exception as e:
        logging.error(f"Error calculating text similarity: {e}")
        return 0.3  # Default score on error

from typing import List, Dict, Optional, Any
import os
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from bson.objectid import ObjectId
from datetime import datetime
import time

# Import dotenv for environment variables
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("db_connector")

class DatabaseConnector:
    def __init__(self):
        # Use environment variable for MongoDB URL with fallback
        self.mongo_url = os.getenv('MONGODB_URL')
        
        # Check if MONGODB_URL is set
        if not self.mongo_url:
            logger.error("MONGODB_URL environment variable not set. Please check your .env file.")
            raise ValueError("MongoDB connection URL not provided")
            
        # Connection retry settings
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        
        # Attempt connection with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Connecting to MongoDB (attempt {attempt+1}/{self.max_retries})")
                self.client = MongoClient(self.mongo_url, serverSelectionTimeoutMS=5000)
                # Check if the connection is working
                self.client.admin.command('ping')
                logger.info("Successfully connected to MongoDB")
                
                # For MongoDB Atlas, we need to use the correct database name
                # Looking at the connection string and the working Node.js backend,
                # we know the database should be in the MongoDB Atlas cluster
                db_name = 'test'  # This is the default database name for MongoDB Atlas
                
                # Try a few common database names used in MongoDB Atlas
                for possible_name in ['test', 'Cluster0', 'main', 'job_portal', 'onjob']:
                    try:
                        temp_db = self.client[possible_name]
                        # Just try to perform a quick operation to check if it works
                        collection_names = temp_db.list_collection_names()
                        logger.info(f"Database '{possible_name}' exists with collections: {collection_names}")
                        
                        # If this has our collections, use it
                        if 'jobs' in collection_names or 'users' in collection_names:
                            db_name = possible_name
                            logger.info(f"Found our collections in database: {db_name}")
                            break
                    except Exception as e:
                        logger.warning(f"Error checking database '{possible_name}': {e}")
                        continue
                logger.info(f"Using database: {db_name}")
                self.db = self.client[db_name]
                
                # Collections (match exact naming from the main project)
                # Log available collections to help debug
                collection_names = self.db.list_collection_names()
                logger.info(f"Available collections: {collection_names}")
                
                # Set up collections with proper error handling
                # These collection names must match exactly what's used in the NodeJS backend
                self.users = self.db.get_collection('users')
                self.jobs = self.db.get_collection('jobs')
                self.job_applications = self.db.get_collection('applications')
                self.feedback = self.db.get_collection('recommendation_feedback')
                
                # Log what we're connecting to
                logger.info(f"MongoDB database: {db_name}")
                logger.info(f"Collections: users, jobs, applications, recommendation_feedback")
                
                # Verify collections exist by attempting to count documents
                try:
                    job_count = self.jobs.count_documents({})
                    user_count = self.users.count_documents({})
                    logger.info(f"Found {job_count} jobs and {user_count} users in the database")                
                except Exception as e:
                    logger.error(f"Error counting documents: {e}")
                
                # Connection successful, break out of retry loop
                break
                
            except (ConnectionFailure, OperationFailure) as e:
                logger.error(f"Failed to connect to MongoDB (attempt {attempt+1}): {e}")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error("Maximum connection attempts reached. Unable to connect to MongoDB.")
                    # Set collections to None to indicate connection failure
                    self.client = None
                    self.db = None
                    self.users = None
                    self.jobs = None
                    self.job_applications = None
                    self.feedback = None
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile from database"""
        try:
            if self.users is None:
                logging.error("MongoDB connection not available")
                return None
                
            # Check if the user_id is a valid ObjectId
            if not ObjectId.is_valid(user_id):
                logging.warning(f"Invalid user_id format: {user_id}")
                return None
                
            user = self.users.find_one({'_id': ObjectId(user_id)})
            if user:
                return {
                    'id': str(user['_id']),
                    'name': user.get('name', ''),
                    'email': user.get('email', ''),
                    'preferred_salary': user.get('preferred_salary', 0),
                    'preferred_location_code': user.get('preferred_location_code', 0),
                    'preferred_location_text': user.get('preferred_location_text', ''),
                    'skills': user.get('skills', []),
                    'profile': user.get('profile', {}),
                    'resume_url': user.get('resume_url', ''),
                    'created_at': user.get('created_at', datetime.now())
                }
            logging.warning(f"User not found: {user_id}")
            return None
        except Exception as e:
            logging.error(f"Error retrieving user profile: {e}")
            return None
    
    def get_all_skills(self) -> List[str]:
        """Get all unique skills from job listings"""
        if self.jobs is None:
            logging.error("MongoDB connection not available")
            return []
            
        try:
            # Collect skills from jobs
            job_skills = set()
            for job in self.jobs.find({}, {'required_skills': 1}):
                if 'required_skills' in job and isinstance(job['required_skills'], list):
                    job_skills.update([skill.lower() for skill in job['required_skills'] if skill])
            
            # Also collect skills from users to improve matching
            user_skills = set()
            if self.users is not None:  # Add check to ensure collection exists
                for user in self.users.find({}, {'skills': 1}):
                    if 'skills' in user and isinstance(user['skills'], list):
                        user_skills.update([skill.lower() for skill in user['skills'] if skill])
            
            # Combine both sets
            all_skills = job_skills.union(user_skills)
            
            return list(all_skills)
        except Exception as e:
            logging.error(f"Error retrieving skills: {e}")
            return []
    
    def get_jobs_by_category(self, category_id: int) -> List[Dict]:
        """Get all jobs in a specific category"""
        if self.jobs is None:
            logging.error("MongoDB connection not available")
            return []
            
        try:
            jobs = self.jobs.find({'category_id': category_id})
            return [self._format_job_document(job) for job in jobs]
        except Exception as e:
            logging.error(f"Error retrieving jobs by category: {e}")
            return []
            
    def get_all_jobs(self) -> List[Dict]:
        """Get all jobs from the database"""
        if self.jobs is None:
            logging.error("MongoDB connection not available")
            return []
            
        try:
            # First check if we have any jobs by counting
            job_count = self.jobs.count_documents({})
            logging.info(f"Found {job_count} jobs in the database")
            
            if job_count == 0:
                # Try looking for jobs with a more general query in case collection structure is different
                all_collections = self.db.list_collection_names()
                logging.info(f"Available collections: {all_collections}")
                
                # Check if we can find a jobs collection with a different name
                job_collections = [coll for coll in all_collections if 'job' in coll.lower()]
                if job_collections:
                    logging.info(f"Found potential job collections: {job_collections}")
                    # Try the first one that might contain jobs
                    alt_jobs = self.db[job_collections[0]]
                    jobs_cursor = alt_jobs.find({}).limit(500)
                    jobs_list = list(jobs_cursor)
                    logging.info(f"Found {len(jobs_list)} jobs in alternate collection {job_collections[0]}")
                    return [self._format_job_document(job) for job in jobs_list]
                
                # If still no jobs, log a sample document from each collection to help debug
                for coll_name in all_collections:
                    try:
                        sample = self.db[coll_name].find_one()
                        if sample:
                            logging.info(f"Sample from {coll_name}: {list(sample.keys())[:5]}")
                    except Exception as e:
                        logging.error(f"Error examining collection {coll_name}: {e}")
                
                logging.error("No jobs found in any collection")
                return []
            
            # Normal path - get jobs from the jobs collection
            jobs_cursor = self.jobs.find({}).limit(500)
            jobs_list = list(jobs_cursor)  # Convert cursor to list to avoid cursor timeout
            logging.info(f"Retrieved {len(jobs_list)} jobs")
            return [self._format_job_document(job) for job in jobs_list]
        except Exception as e:
            logging.error(f"Error retrieving all jobs: {e}")
            return []
    
    def get_all_job_descriptions(self) -> List[str]:
        """Get all job descriptions for text analysis"""
        if self.jobs is None:
            logging.error("MongoDB connection not available")
            return []
            
        try:
            descriptions = []
            for job in self.jobs.find({}, {'description': 1}):
                if 'description' in job and job['description']:
                    descriptions.append(job['description'])
            return descriptions
        except Exception as e:
            logging.error(f"Error retrieving job descriptions: {e}")
            return []
            
    def _format_job_document(self, job: Dict) -> Dict:
        """Format a job document for API response"""
        # Create a copy to avoid modifying the original
        formatted_job = {}
        
        # Extract _id and convert to string
        if '_id' in job:
            formatted_job['id'] = str(job['_id'])
        
        # Copy important fields directly from the job document
        # Use the actual field names from your MongoDB job sample
        formatted_job['title'] = job.get('Title', '')
        formatted_job['company'] = job.get('Company', '')
        formatted_job['location'] = job.get('Location', '')
        formatted_job['category'] = job.get('Category', '')
        
        # Handle job description
        description = ""
        if 'job_description' in job and isinstance(job['job_description'], dict):
            # Combine overview and responsibilities/requirements
            if 'overview' in job['job_description']:
                description += job['job_description']['overview'] + "\n\n"
            
            if 'responsibilities' in job['job_description'] and isinstance(job['job_description']['responsibilities'], list):
                description += "Responsibilities:\n"
                description += "\n".join([f"- {resp}" for resp in job['job_description']['responsibilities']]) + "\n\n"
            
            if 'requirements' in job['job_description'] and isinstance(job['job_description']['requirements'], list):
                description += "Requirements:\n"
                description += "\n".join([f"- {req}" for req in job['job_description']['requirements']])
                
        formatted_job['description'] = description
        
        # Handle skills
        required_skills = []
        preferred_skills = []
        
        if 'skills' in job and isinstance(job['skills'], dict):
            if 'required' in job['skills'] and isinstance(job['skills']['required'], list):
                required_skills = job['skills']['required']
            if 'preferred' in job['skills'] and isinstance(job['skills']['preferred'], list):
                preferred_skills = job['skills']['preferred']
        
        formatted_job['required_skills'] = required_skills
        formatted_job['preferred_skills'] = preferred_skills
        
        # Handle salary
        salary_min = 0
        salary_max = 0
        if 'salary' in job:
            if isinstance(job['salary'], int):
                salary_min = job['salary']
                salary_max = job['salary'] * 1.2  # Estimate max as 20% higher than minimum
            elif isinstance(job['salary'], dict) and 'min' in job['salary'] and 'max' in job['salary']:
                salary_min = job['salary']['min']
                salary_max = job['salary']['max']
        
        formatted_job['salary_range'] = {
            'min': salary_min,
            'max': salary_max,
            'display': job.get('salaryDisplay', f"{salary_min}-{salary_max}")
        }
        
        # Additional metadata
        formatted_job['employment_type'] = job.get('jobType', 'Full-time')
        formatted_job['workplace_type'] = job.get('workplaceType', 'Onsite')
        formatted_job['experience_level'] = job.get('experience', 'Entry Level')
        
        # Experience years
        if 'experienceYears' in job and isinstance(job['experienceYears'], dict):
            formatted_job['experience_years'] = {
                'min': job['experienceYears'].get('min', 0),
                'max': job['experienceYears'].get('max', 0)
            }
        
        # Tags for enhanced matching
        formatted_job['tags'] = job.get('tags', [])
        
        # Date information
        formatted_job['date_posted'] = job.get('postedAt', '')
        formatted_job['application_deadline'] = job.get('applicationDeadline', '')
        
        return formatted_job
    
    def get_all_jobs(self):
        """Get all jobs from the database"""
        if not self.jobs:
            logger.error("MongoDB connection not available")
            return []
            
        try:
            logger.info("Retrieving all jobs from database")
            
            # Get jobs from database
            cursor = self.jobs.find({})
            
            # Process documents
            jobs = []
            for job in cursor:
                try:
                    # Format job document for API response
                    formatted_job = self._format_job_document(job)
                    jobs.append(formatted_job)
                except Exception as e:
                    # If there's an error formatting a specific job, log it but continue
                    logger.warning(f"Error formatting job document: {e}")
                    continue
            
            logger.info(f"Retrieved {len(jobs)} jobs from database")
            return jobs
        except Exception as e:
            logger.error(f"Error retrieving jobs: {e}")
            return []
            
    async def get_jobs(self, limit=100, skip=0, filters=None):
        """Get jobs with pagination and filtering support - for Hugging Face API integration"""
        if not self.jobs:
            logger.error("MongoDB connection not available")
            return []
            
        try:
            logger.info(f"Retrieving jobs from database (limit={limit}, skip={skip})")
            
            # Prepare query filters
            query = {}
            if filters:
                # Apply any filters (e.g., category, location, etc.)
                if 'category' in filters and filters['category']:
                    query['Category'] = filters['category']
                if 'location' in filters and filters['location']:
                    query['Location'] = {'$regex': filters['location'], '$options': 'i'}
            
            # Get jobs from database with pagination
            cursor = self.jobs.find(query).skip(skip).limit(limit)
            
            # Process documents
            jobs = []
            for job in cursor:
                try:
                    # Format job document for API response
                    formatted_job = self._format_job_document(job)
                    jobs.append(formatted_job)
                except Exception as e:
                    # If there's an error formatting a specific job, log it but continue
                    logger.warning(f"Error formatting job document: {e}")
                    continue
            
            logger.info(f"Retrieved {len(jobs)} jobs from database")
            return jobs
        except Exception as e:
            logger.error(f"Error retrieving jobs: {e}")
            return []
    
    def store_recommendation_feedback(self, user_id: str, job_id: str, rating: int, 
    timestamp: datetime = None) -> bool:
        """Store user feedback on job recommendations"""
        if not self.feedback:
            logging.error("MongoDB connection not available")
            return False
            
        try:
            if not timestamp:
                timestamp = datetime.now()
                
            # Validate rating
            rating = max(1, min(5, rating))  # Ensure rating is between 1-5
            
            # Insert feedback
            result = self.feedback.insert_one({
                'user_id': ObjectId(user_id),
                'job_id': ObjectId(job_id),
                'rating': rating,
                'timestamp': timestamp
            })
            
            return result.acknowledged
        except Exception as e:
            logging.error(f"Error storing recommendation feedback: {e}")
            return False
    
    def record_job_application(self, user_id: str, job_id: str, status: str = 'applied') -> bool:
        """Record a job application from a user"""
        if self.job_applications is None:
            logging.error("MongoDB connection not available")
            return False
            
        try:
            # Insert application record
            result = self.job_applications.insert_one({
                'user_id': ObjectId(user_id),
                'job_id': ObjectId(job_id),
                'status': status,
                'applied_at': datetime.now(),
                'updated_at': datetime.now()
            })
            
            return result.acknowledged
        except Exception as e:
            logging.error(f"Error recording job application: {e}")
            return False
            
    def close(self):
        """Close the MongoDB connection"""
        if self.client:
            self.client.close()
            logging.info("MongoDB connection closed")
        else:
            logging.warning("No MongoDB connection to close")
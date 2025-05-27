"""
Initialize the AI model with sample data and start the server.
This script will:
1. Check the database connection
2. Insert a sample job if needed
3. Train the model
4. Start the FastAPI server
"""
import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv
from bson.objectid import ObjectId
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def ensure_sample_job_exists():
    """Make sure at least one job exists in the database for training"""
    try:
        # Connect to MongoDB
        mongo_url = os.getenv('MONGODB_URL')
        if not mongo_url:
            logger.error("MONGODB_URL not found in .env file")
            return False
        
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
        
        # Try to find the database with our collections
        db = None
        for db_name in ['test', 'Cluster0', 'main', 'job_portal', 'onjob']:
            try:
                temp_db = client[db_name]
                collections = temp_db.list_collection_names()
                
                if 'jobs' in collections:
                    db = temp_db
                    logger.info(f"Found jobs collection in database: {db_name}")
                    break
            except Exception:
                continue
        
        if db is None:
            # If no database found with jobs collection, use default
            db = client['test']
            logger.info("No database found with jobs collection, using 'test' database")
            
        # Check if jobs collection exists and has data
        jobs = db.jobs
        job_count = jobs.count_documents({})
        
        if job_count == 0:
            logger.info("No jobs found in database, inserting sample job...")
            
            # Insert a sample job based on the example provided
            sample_job = {
                "Company": "Microsoft",
                "Title": "Senior Software Engineer",
                "Category": "Software Development",
                "Location": "Bangalore",
                "workplaceType": "Hybrid",
                "jobType": "Full-time",
                "experience": "Senior Level",
                "experienceYears": {
                    "min": 5,
                    "max": 8
                },
                "salary": 200000,
                "salaryDisplay": "₹200,000-250,000 LPA",
                "benefits": [
                    "Health Insurance",
                    "Life Insurance",
                    "401(k)/Retirement",
                    "Paid Time Off",
                    "Stock Options",
                    "Mental Health Benefits",
                    "Education Reimbursement"
                ],
                "skills": {
                    "required": [
                        "C#",
                        ".NET Core",
                        "Azure",
                        "Microservices",
                        "REST APIs"
                    ],
                    "preferred": [
                        "Kubernetes",
                        "Docker",
                        "CI/CD",
                        "GraphQL"
                    ]
                },
                "qualifications": {
                    "education": "Bachelor's",
                    "additionalCertifications": [
                        "Microsoft Certified",
                        "Azure Solutions Architect"
                    ]
                },
                "job_description": {
                    "overview": "Join Microsoft's Cloud & AI division to build next-generation cloud services and developer tools.",
                    "responsibilities": [
                        "Design and implement scalable microservices",
                        "Lead technical design discussions and architecture reviews",
                        "Mentor junior developers and promote best practices",
                        "Collaborate with cross-functional teams globally",
                        "Drive innovation in cloud services"
                    ],
                    "requirements": [
                        "5+ years of experience in software development",
                        "Strong expertise in C# and .NET Core",
                        "Experience with cloud platforms, preferably Azure",
                        "Excellent problem-solving and communication skills"
                    ]
                },
                "applicationDeadline": datetime.now(),
                "numberOfOpenings": 3,
                "applicationProcess": "Direct",
                "company": {
                    "description": "Microsoft Corporation is a technology company that develops and supports software, services, and devices.",
                    "website": "https://microsoft.com",
                    "industry": "Technology",
                    "companySize": "1000+",
                    "fundingStage": "Public"
                },
                "status": "Active",
                "tags": [
                    "software development",
                    "cloud computing",
                    "azure",
                    ".net",
                    "senior engineer"
                ],
                "applicationStatistics": {
                    "views": 2300,
                    "applications": 180,
                    "lastViewedAt": datetime.now()
                },
                "postedAt": datetime.now(),
                "companyLogo": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Microsoft_logo.svg/2048px-Microsoft_logo.svg.png"
            }
            
            result = jobs.insert_one(sample_job)
            logger.info(f"Inserted sample job with ID: {result.inserted_id}")
            
            # Insert a second sample job with different skills
            sample_job2 = {
                "Company": "Google",
                "Title": "Machine Learning Engineer",
                "Category": "Artificial Intelligence",
                "Location": "Remote",
                "workplaceType": "Remote",
                "jobType": "Full-time",
                "experience": "Mid Level",
                "experienceYears": {
                    "min": 3,
                    "max": 5
                },
                "salary": 180000,
                "salaryDisplay": "₹180,000-220,000 LPA",
                "benefits": [
                    "Health Insurance",
                    "Paid Time Off",
                    "Stock Options",
                    "Education Reimbursement"
                ],
                "skills": {
                    "required": [
                        "Python",
                        "TensorFlow",
                        "Machine Learning",
                        "Data Science",
                        "SQL"
                    ],
                    "preferred": [
                        "PyTorch",
                        "Cloud ML",
                        "NLP",
                        "Computer Vision"
                    ]
                },
                "qualifications": {
                    "education": "Master's",
                    "additionalCertifications": [
                        "Google Cloud Certified",
                        "TensorFlow Developer"
                    ]
                },
                "job_description": {
                    "overview": "Join Google's AI team to develop cutting-edge machine learning solutions.",
                    "responsibilities": [
                        "Design and implement machine learning models",
                        "Analyze large datasets for insights",
                        "Deploy models to production environments",
                        "Optimize model performance and accuracy",
                        "Stay updated with latest AI research"
                    ],
                    "requirements": [
                        "3+ years of experience in machine learning",
                        "Strong Python programming skills",
                        "Experience with TensorFlow or PyTorch",
                        "Degree in Computer Science, Mathematics, or related field"
                    ]
                },
                "applicationDeadline": datetime.now(),
                "numberOfOpenings": 2,
                "applicationProcess": "Direct",
                "company": {
                    "description": "Google is a technology company specializing in search, cloud computing, and AI.",
                    "website": "https://google.com",
                    "industry": "Technology",
                    "companySize": "10000+",
                    "fundingStage": "Public"
                },
                "status": "Active",
                "tags": [
                    "machine learning",
                    "artificial intelligence",
                    "python",
                    "data science",
                    "tensorflow"
                ],
                "applicationStatistics": {
                    "views": 1800,
                    "applications": 150,
                    "lastViewedAt": datetime.now()
                },
                "postedAt": datetime.now(),
                "companyLogo": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Google_%22G%22_Logo.svg/2048px-Google_%22G%22_Logo.svg.png"
            }
            
            result2 = jobs.insert_one(sample_job2)
            logger.info(f"Inserted second sample job with ID: {result2.inserted_id}")
            
            return True
        else:
            logger.info(f"Found {job_count} existing jobs in the database")
            return True
    
    except Exception as e:
        logger.error(f"Error ensuring sample job exists: {e}")
        return False

def train_model():
    """Train the recommendation model"""
    try:
        logger.info("Training the recommendation model...")
        
        # Import the recommender
        from recommendation_model import JobRecommender
        
        # Create and train the model
        recommender = JobRecommender()
        success = recommender.create_and_train_model()
        
        if success:
            logger.info("✅ Successfully trained the recommendation model")
            return True
        else:
            logger.error("❌ Failed to train the recommendation model")
            return False
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    try:
        logger.info("Starting the FastAPI server...")
        
        # Run the app.py file using Python
        subprocess.Popen([sys.executable, "app.py"])
        
        logger.info("✅ Server started successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return False

if __name__ == "__main__":
    try:
        # Step 1: Ensure sample job exists
        if ensure_sample_job_exists():
            # Step 2: Train the model
            if train_model():
                # Step 3: Start the server
                start_server()
                
                logger.info("AI Model system initialized and started successfully!")
                logger.info("You can now use the AI model for job recommendations.")
                logger.info("API is available at: http://127.0.0.1:8000")
                logger.info("Swagger docs at: http://127.0.0.1:8000/api/docs")
                
        else:
            logger.error("Failed to initialize the system")
    
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")

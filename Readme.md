# OnJob AI Model - Advanced Job Recommendation System

## Overview
This AI model powers the OnJob platform with advanced job recommendation capabilities. It analyzes user resumes and job listings to deliver personalized job recommendations based on skills, experience, and other relevant factors. The system combines machine learning, natural language processing, and rule-based matching to ensure high-quality matches even with limited data.

## Features
- **AI-powered job matching**: Uses TensorFlow to learn patterns between user skills and job requirements
- **Skill extraction and analysis**: Identifies relevant skills from both resumes and job listings
- **Fallback mechanisms**: Ensures recommendations are provided even in edge cases
- **MongoDB integration**: Connects directly to the same database used by the main application
- **RESTful API**: Provides easy integration with the Node.js backend
- **Real-time recommendations**: Processes and returns results within seconds

## Tech Stack
- **Framework**: FastAPI (Python-based high-performance API)
- **AI/ML**: TensorFlow, scikit-learn, NLTK
- **Database**: MongoDB (shared with main application)
- **Text Processing**: TF-IDF Vectorization, Skill Extraction
- **Deployment**: Docker-ready for easy deployment

---

## Getting Started

### Prerequisites
Ensure you have Python 3.9+ installed along with the required dependencies:

```sh
pip install -r requirements.txt
```

### Environment Setup
Configure your environment variables in the `.env` file:
```
# MongoDB Connection
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority

# API Security
API_KEY=your_api_key  # Same key used in Node.js backend

# App Settings
PORT=8000
ENV=development
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

## Quick Start

Use the provided initialization script to start the recommendation system:

```sh
python initialize_and_start.py
```

This script will:
1. Verify database connection
2. Create and train the AI model if needed
3. Start the FastAPI server

## API Endpoints

### `GET /`
Health check endpoint to verify the API is running.

### `POST /api/analyze-resume`
Analyzes a resume and extracts skills, experience, and education.

### `POST /api/recommend-jobs`
Provides job recommendations based on user profile and resume data.

### `POST /api/recommendation-feedback`
Collects user feedback on recommendations for future model improvement.

## How the AI Model Works

1. **Data Collection**: Connects to MongoDB to retrieve job listings and user profiles
2. **Feature Extraction**: 
   - Converts job descriptions to TF-IDF vectors
   - Creates skill match vectors between user skills and job requirements
   - Extracts numeric features like salary and experience years
3. **Model Training**: Trains a neural network to predict job-candidate matches
4. **Scoring**: Ranks jobs based on match scores and similarity metrics
5. **Recommendation**: Returns a personalized list of jobs with match explanations

## Resilient Design

The system includes multiple fallback mechanisms:
- If the ML model fails, it uses a rule-based matching algorithm
- If skill extraction fails, it adds default skills to ensure recommendations
- If no matches are found with strict criteria, it loosens requirements

## Integration with Node.js Backend

The Node.js backend communicates with this AI model through a REST API. The backend sends user profile data and receives ranked job recommendations. This separation of concerns allows the AI model to be developed and scaled independently.

---

## Directory Structure
```
AI-Model/
│── app.py                   # FastAPI application entry point
│── recommendation_model.py   # ML model implementation
│── initialize_and_start.py   # Initialization and startup script
│
│── database/
│   └── db_connector.py      # MongoDB connection and data access
│
│── models/
│   ├── __init__.py
│   ├── bert_embedding.py    # Text embedding utilities
│   ├── recommendation_model.py  # Core ML model architecture
│   └── job_recommender_model.keras  # Trained model (generated)
│
│── services/
│   ├── job_matcher.py       # Job matching algorithms
│   ├── job_matcher_helper.py  # Helper utilities for matching
│   └── resume_parser.py     # Resume text extraction and analysis
│
│── requirements.txt         # Python dependencies
└── .env                     # Environment configuration
```

---

## License
Proprietary - OnJob © 2025

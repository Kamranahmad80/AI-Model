#!/bin/bash
# Download NLTK data needed for the application
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Start the application
uvicorn app:app --host 0.0.0.0 --port $PORT

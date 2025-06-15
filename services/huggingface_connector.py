import os
import requests
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("huggingface_connector")

class HuggingFaceConnector:
    def __init__(self):
        self.token = os.getenv("HUGGINGFACE_TOKEN")
        self.model_id = os.getenv("HUGGINGFACE_MODEL_ID", "kamranahmad80/job-recommendation")
        self.api_url = f"{os.getenv('HUGGINGFACE_API_URL', 'https://api-inference.huggingface.co/models/')}{self.model_id}"
        self.headers = {"Authorization": f"Bearer {self.token}"}
        logger.info(f"Initialized HuggingFace connector for model: {self.model_id}")
        
    def get_recommendations(self, resume_data, jobs_data, user_profile=None):
        """
        Get job recommendations from Hugging Face model
        """
        try:
            # Prepare payload
            payload = {
                "inputs": {
                    "resume": resume_data,
                    "jobs": jobs_data,
                    "user_profile": user_profile if user_profile else {}
                }
            }
            
            # Make API request
            logger.info(f"Sending request to HuggingFace API for model: {self.model_id}")
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            # Check for errors
            if response.status_code != 200:
                logger.error(f"HuggingFace API error: {response.status_code} - {response.text}")
                return []
                
            # Parse response
            result = response.json()
            
            # The response structure will depend on your specific model
            # Adjust the parsing logic based on your model's output format
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "recommendations" in result:
                return result["recommendations"]
            else:
                logger.warning(f"Unexpected response format from HuggingFace: {result}")
                return []
                
        except Exception as e:
            logger.error(f"Error connecting to HuggingFace API: {str(e)}")
            return []

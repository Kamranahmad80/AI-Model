import PyPDF2
import docx
import re
import os
import json
import requests
from typing import List, Dict, Set
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.chunk import RegexpParser

# Download required NLTK data with error handling
def download_nltk_data():
    nltk_resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
    
    for resource in nltk_resources:
        try:
            # Check if the resource is already downloaded
            if not nltk.data.find(f'tokenizers/{resource}')\
               and not nltk.data.find(f'corpora/{resource}')\
               and not nltk.data.find(f'taggers/{resource}')\
               and not nltk.data.find(f'chunkers/{resource}'):
                nltk.download(resource, quiet=True)
        except (LookupError, PermissionError, OSError) as e:
            print(f"Warning: Could not download NLTK resource '{resource}': {e}")
            print(f"The system will try to operate without this resource.")
            
try:
    download_nltk_data()
except Exception as e:
    print(f"Warning: NLTK data download issue: {e}")
    print("The system will attempt to continue with available resources.")

class ResumeParser:
    def __init__(self):
        # Load comprehensive skills database
        self.skills_keywords = self._load_skills_database()
        
        # Common skill phrases for regex matching
        self.skill_patterns = [
            r'\b(?:proficient|experienced|skilled|expertise)\s+in\s+([^.,:;!?\n]+)',
            r'\b(?:knowledge|understanding)\s+of\s+([^.,:;!?\n]+)',
            r'\bskills\s*:([^.,:;!?\n]+)',
            r'\btechnologies\s*:([^.,:;!?\n]+)',
            r'\blanguages\s*:([^.,:;!?\n]+)'
        ]
        
        # Education related keywords for NER
        self.education_keywords = ['degree', 'bachelor', 'master', 'phd', 'diploma', 
                              'university', 'college', 'school', 'institute', 'education']
    
    def _load_skills_database(self) -> Set[str]:
        """Load or create a comprehensive skills database"""
        skills_file = os.path.join(os.path.dirname(__file__), '../dataset/skills_database.json')
        
        if os.path.exists(skills_file):
            with open(skills_file, 'r') as f:
                return set(json.load(f))
        else:
            # Base tech skills list as a fallback
            base_skills = {
                # Programming languages
                "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "php", "swift", "kotlin",
                "go", "rust", "scala", "perl", "r", "matlab", "bash", "powershell", "sql", "nosql",
                
                # Frameworks & Libraries
                "react", "angular", "vue.js", "node.js", "express", "django", "flask", "spring", "asp.net",
                "laravel", "tensorflow", "pytorch", "keras", "pandas", "numpy", "scikit-learn", "matplotlib",
                
                # Databases
                "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "cassandra", "oracle", "sql server",
                
                # Cloud & DevOps
                "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform", "ansible", "ci/cd",
                "git", "github", "gitlab", "bitbucket",
                
                # Soft Skills
                "project management", "team leadership", "communication", "problem solving", "critical thinking",
                "time management", "agile", "scrum", "kanban", "jira", "trello",
                
                # Data Science & Analytics
                "machine learning", "deep learning", "data analysis", "data visualization", "data mining",
                "statistics", "big data", "hadoop", "spark", "tableau", "power bi", "data modeling"
            }
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(skills_file), exist_ok=True)
            
            # Save the base skills to file
            with open(skills_file, 'w') as f:
                json.dump(list(base_skills), f)
            
            return base_skills
    
    async def parse_resume(self, file) -> str:
        """Parse resume file and return text content"""
        content = await file.read()
        file_ext = file.filename.split('.')[-1].lower()
        
        if file_ext == 'pdf':
            return self._parse_pdf(content)
        elif file_ext in ['docx', 'doc']:
            return self._parse_docx(content)
        else:
            raise ValueError("Unsupported file format")
    
    def _parse_pdf(self, content: bytes) -> str:
        """Extract text from PDF"""
        pdf_reader = PyPDF2.PdfReader(content)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    def _parse_docx(self, content: bytes) -> str:
        """Extract text from DOCX"""
        doc = docx.Document(content)
        return " ".join([paragraph.text for paragraph in doc.paragraphs])
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text using multiple techniques"""
        text_lower = text.lower()
        skills = set()
        
        # 1. Direct matching with skill keywords
        for skill in self.skills_keywords:
            if f" {skill} " in f" {text_lower} " or f",{skill}," in f",{text_lower},":
                skills.add(skill)
        
        # 2. NLP-based extraction
        sentences = sent_tokenize(text)
        for sentence in sentences:
            # Apply POS tagging
            tokens = word_tokenize(sentence)
            tagged = nltk.pos_tag(tokens)
            
            # Find noun phrases that might be skills
            grammar = "NP: {<JJ>*<NN>+}"
            chunk_parser = RegexpParser(grammar)
            chunks = chunk_parser.parse(tagged)
            
            for subtree in chunks.subtrees(filter=lambda t: t.label() == 'NP'):
                potential_skill = ' '.join(word for word, tag in subtree.leaves())
                if potential_skill.lower() in self.skills_keywords:
                    skills.add(potential_skill.lower())
        
        # 3. Regex pattern matching for skill sections
        for pattern in self.skill_patterns:
            matches = re.finditer(pattern, text_lower, re.MULTILINE)
            for match in matches:
                skill_section = match.group(1).strip()
                # Split by common separators
                for separator in [',', ';', '•', '\n', '|', '/']:
                    if separator in skill_section:
                        for skill in skill_section.split(separator):
                            potential_skill = skill.strip().lower()
                            # Check if this is in our skills database
                            if potential_skill in self.skills_keywords:
                                skills.add(potential_skill)
                            # Check for multi-word skills
                            for known_skill in self.skills_keywords:
                                if " " in known_skill and known_skill in potential_skill:
                                    skills.add(known_skill)
        
        return list(skills)
    
    def extract_experience(self, text: str) -> List[Dict]:
        """Extract work experience from resume text using improved patterns"""
        # More comprehensive regex patterns for experience extraction
        patterns = [
            # Standard format: 2020 - 2023 Company Name
            r"(\d{4})\s*(?:-|to|–)\s*(\d{4}|present|current|now)\s*(.+?)(?:\n|$)",
            
            # Company then date: Company Name (2020 - 2023)
            r"(.+?)\s*\(\s*(\d{4})\s*(?:-|to|–)\s*(\d{4}|present|current|now)\s*\)",
            
            # Month Year format: Jan 2020 - Mar 2023 Company Name
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})\s*(?:-|to|–)\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4}|present|current|now)\s*(.+?)(?:\n|$)"
        ]
        
        experiences = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                
                # Handle different patterns with different group positions
                if len(groups) == 3:
                    if groups[0].isdigit():  # First pattern: year-year-company
                        start_year, end_year, description = groups
                    else:  # Second pattern: company-year-year
                        description, start_year, end_year = groups
                
                # Normalize the end year
                if end_year.lower() in ('present', 'current', 'now'):
                    end_year = 'present'
                
                # Extract company and role if possible
                company = description.strip()
                role = ""
                
                # Try to separate role from company if they're in the format "Role at Company"
                if ' at ' in company:
                    parts = company.split(' at ')
                    role = parts[0].strip()
                    company = parts[1].strip()
                
                experiences.append({
                    "start_year": start_year,
                    "end_year": end_year,
                    "company": company,
                    "role": role,
                    "description": description.strip()
                })
        
        # De-duplicate by start/end year and company
        unique_experiences = []
        seen = set()
        
        for exp in experiences:
            key = (exp['start_year'], exp['end_year'], exp['company'])
            if key not in seen:
                seen.add(key)
                unique_experiences.append(exp)
        
        return unique_experiences
        
    def extract_education(self, text: str) -> List[Dict]:
        """Extract education information from resume"""
        education_info = []
        
        # Pattern for education entries
        edu_patterns = [
            # Degree from University (Year-Year)
            r"([^\n.]+(?:degree|bachelor|master|phd|diploma|certificate)[^\n.]+)\s+from\s+([^\n.]+)\s*\(?\s*(\d{4})\s*(?:-|to|–)\s*(\d{4}|present|current|now)\s*\)?",
            
            # University (Year-Year) - Degree
            r"([^\n.]+(?:university|college|institute|school)[^\n.]+)\s*\(?\s*(\d{4})\s*(?:-|to|–)\s*(\d{4}|present|current|now)\s*\)?\s*[-–]?\s*([^\n.]+(?:degree|bachelor|master|phd|diploma)[^\n.]+)"
        ]
        
        for pattern in edu_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                
                if len(groups) == 4:
                    if 'university' in groups[0].lower() or 'college' in groups[0].lower():
                        # Second pattern: university-year-year-degree
                        institution, start_year, end_year, degree = groups
                    else:
                        # First pattern: degree-university-year-year
                        degree, institution, start_year, end_year = groups
                    
                    education_info.append({
                        "degree": degree.strip(),
                        "institution": institution.strip(),
                        "start_year": start_year,
                        "end_year": end_year if end_year.lower() not in ('present', 'current', 'now') else 'present'
                    })
        
        return education_info
# File: scripts/preprocess_jobs.py
import pandas as pd
import string

def clean_text(text):
    """Converts text to lowercase and removes punctuation."""
    if isinstance(text, str):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    return ""

def preprocess_jobs(input_path="../dataset/jobs.csv", output_path="../dataset/jobs_cleaned.csv"):
    # Load jobs dataset
    jobs_df = pd.read_csv(input_path)
    # Merge columns into a new column 'job_description'
    jobs_df['job_description'] = jobs_df[['Responsibilities', 'Minimum Qualifications', 'Preferred Qualifications']]\
        .fillna('').agg(' '.join, axis=1)
    # Clean the merged job description
    jobs_df['cleaned_description'] = jobs_df['job_description'].apply(clean_text)
    # Save the cleaned data for future use
    jobs_df.to_csv(output_path, index=False)
    print("Jobs data preprocessed and saved to", output_path)
    return jobs_df

if __name__ == "__main__":
    preprocess_jobs()

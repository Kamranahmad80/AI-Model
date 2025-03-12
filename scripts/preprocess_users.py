# File: scripts/preprocess_users.py
import pandas as pd
import string

def clean_text(text):
    """Converts text to lowercase and removes punctuation."""
    if isinstance(text, str):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    return ""

def preprocess_users(input_path="../dataset/users.csv", output_path="../dataset/users_cleaned.csv"):
    user_df = pd.read_csv(input_path)
    # Check what columns are available (adjust 'Skills' if necessary)
    print("User CSV columns:", user_df.columns.tolist())
    # Clean the skills column
    user_df['cleaned_skills'] = user_df['Skills'].apply(clean_text)
    # Save the cleaned data
    user_df.to_csv(output_path, index=False)
    print("Users data preprocessed and saved to", output_path)
    return user_df

if __name__ == "__main__":
    preprocess_users()

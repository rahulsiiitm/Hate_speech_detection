import re
import string
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans raw text data.
    1. Lowercases text.
    2. Removes URLs and punctuation.
    3. Removes stop words to reduce noise.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs (http/https/www)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 4. Remove Stop Words
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    
    return " ".join(filtered_words)

def process_dataset(input_filepath, output_filepath):
    """
    Loads raw data, applies cleaning, and saves to processed folder.
    """
    # 1. Check if file exists
    if not os.path.exists(input_filepath):
        print(f"Error: File not found at {input_filepath}")
        return

    print(f"Loading raw data from {input_filepath}...")
    df = pd.read_csv(input_filepath)
    
    # 2. Apply cleaning
    print("Cleaning text...")
    # Use 'tweet' column if it exists (Kaggle dataset), otherwise 'text'
    text_col = 'tweet' if 'tweet' in df.columns else 'text'
    
    if text_col not in df.columns:
        print(f"Error: Column '{text_col}' not found in dataset.")
        return

    # Create a new column for cleaned text
    df['cleaned_text'] = df[text_col].apply(clean_text)
    
    # 3. Save to new CSV
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    # We save only the necessary columns to keep it clean
    # Keeping 'class' (label) and 'cleaned_text'
    output_df = df[['class', 'cleaned_text']] if 'class' in df.columns else df[['cleaned_text']]
    
    output_df.to_csv(output_filepath, index=False)
    print(f"Success! Processed data saved to {output_filepath}")

if __name__ == "__main__":
    # Define paths relative to the project root
    raw_data_path = os.path.join('data', 'raw', 'labeled_data.csv')
    processed_data_path = os.path.join('data', 'processed', 'processed_data.csv')
    
    process_dataset(raw_data_path, processed_data_path)
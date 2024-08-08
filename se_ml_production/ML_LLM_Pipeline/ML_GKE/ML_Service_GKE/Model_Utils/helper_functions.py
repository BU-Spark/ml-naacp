import re
import json

from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

# Helper functions for the model training and prediction pipeline

# Format the dataframe 
def clean_df(df):
    df = pd.concat([df['content_id'], df['Headline'], df['Body']], axis=1) # We just need the ID, Header, and Body to run the pipeline

    # Remove duplicates based on Headers
    duplicates = df.duplicated(subset=['Headline']) 
    print(f"[INFO] Duplicates in DF:\n {duplicates.value_counts()}")
    df = df.drop_duplicates(subset=['Headline'])

    # Clean the HTML in the Body and header -> Regex Cleaner 
    func_clean_html = lambda x: BeautifulSoup(x, "html.parser").get_text() # HTML Cleaner
    df['Body'] = df['Body'].progress_apply(func_clean_html)
    df['Headline'] = df['Headline'].progress_apply(func_clean_html)
    func_clean_regex = lambda x: ' '.join([item for item in re.findall(r'[A-Za-z0-9!@#$%^&*().]+', x) if len(item) > 1]) # Regex Cleaner
    df['Body'] = df['Body'].progress_apply(func_clean_regex)
    df['Headline'] = df['Headline'].progress_apply(func_clean_regex)

    return df

# Load the cache from the file
def load_cache(path):
    try:
        with open(path, 'r') as file:
            cache = json.load(file)
    except FileNotFoundError:
        cache = {}
    return cache

# Save cache to file
def save_cache(cache, path):
    with open(path, 'w') as file:
        json.dump(cache, file, indent=4)

# Wrapper function to measure time taken by a given function
import time

def check_time(func):
    def sec_to_hms(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remaining_seconds = round(seconds % 60)
        return f"{hours:02}:{minutes:02}:{remaining_seconds:02}"
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        total_time_formatted = sec_to_hms(total_time)
        print(f"Time taken: {total_time_formatted}")
        return result
    return wrapper
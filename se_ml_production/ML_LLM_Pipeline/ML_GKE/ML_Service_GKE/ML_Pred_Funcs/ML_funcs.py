import re
import json
import pandas as pd
from bs4 import BeautifulSoup

from tqdm import tqdm
tqdm.pandas()

from global_state import global_instance
from Model_Utils.model_Utils import explicit_filtering, explicit_filtering_NER, filter_loc_explicit, getLongLatsForFAC, predict_llama, predict_NER_def, format_NER, filter_loc, getLongLats, getTractList

import os
import tiktoken
import numpy as np
from transformers import pipeline
from sklearn.metrics import adjusted_rand_score
from openai import OpenAI, AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, wait_random_exponential, stop_after_attempt


# ====== TOPIC MODELING PIPELINE ======
def truncate(tokens, length=500):
    """
    Function to get the first 500 elements from a list
    """
    return tokens[:length]

def topic_modeling(df):
    """
    Processes dataframe and passes it to output topic labels. Does Topic Modeling task on articles.
    
    Parameters
    ----
    df: The pandas dataframe that topic modeling is being done on.

    Returns
    ---- 
    Returns a Dataframe of Topic Modeling articles
    """
    try:
        df['topic_model_body'] = df['Body'].progress_apply(lambda x: re.sub(re.compile('<.*?>'), '', x))
        df['tokens'] = df['topic_model_body'].progress_apply(lambda x: x.split())
        df['tokens'] = df['tokens'].progress_apply(truncate)
        df['ada_embedding'] = df.tokens.apply(lambda x: global_instance.get_data('openAIClient').get_embedding(','.join(map(str,x)), model='text-embedding-3-small'))

        # Find most similar taxonomy (out of all toipcs) to news body
        closest_topic_list_all = []
        for index, row in df.iterrows():
            target_embedding = row['ada_embedding']
            similarities = [cosine_similarity(np.array(target_embedding).reshape(1, -1), np.array(topic).reshape(1, -1))[0][0] for topic in global_instance.get_data("all_topics_embedding")]

            # Find the index of the topic with the highest similarity
            closest_topic_index = np.argmax(similarities)

            # Retrieve the closest topic embedding
            closest_topic = global_instance.get_data("all_topics_list")[closest_topic_index]
            closest_topic_list_all.append(closest_topic)
        df['closest_topic_all'] = closest_topic_list_all

        closest_topic_list_selected = []
        for index, row in df.iterrows():
            target_embedding = row['ada_embedding']
            similarities = [cosine_similarity(np.array(target_embedding).reshape(1, -1), np.array(topic).reshape(1, -1))[0][0] for topic in global_instance.get_data("selected_topics_embedding")]

            # Find the index of the topic with the highest similarity
            closest_topic_index = np.argmax(similarities)

            # Retrieve the closest topic embedding
            closest_topic = global_instance.get_data("selected_topics_list")[closest_topic_index]
            closest_topic_list_selected.append(closest_topic)

        df['closest_topic_selected'] = closest_topic_list_selected

        client_topic_embedding_list = global_instance.get_data("client_taxonomy_df")['ada_embedding'].to_list()
        client_topic_list = global_instance.get_data("client_taxonomy_df")['label'].to_list()
        similarity_arr = []

        closest_topic_list_client = []
        for index, row in df.iterrows():
            target_embedding = row['ada_embedding']
            similarities = [cosine_similarity(np.array(target_embedding).reshape(1, -1), np.array(topic).reshape(1, -1))[0][0] for topic in client_topic_embedding_list]
            
            if max(similarities) > 0.25:    
                closest_topic_index = np.argmax(similarities) # Find the index of the topic with the highest similarity
                closest_topic = client_topic_list[closest_topic_index] # Retrieve the closest topic embedding
                closest_topic_list_client.append(closest_topic)
            else:
                closest_topic_list_client.append('Other')
            similarity_arr.append(max(similarities))
            
        df['closest_topic_client'] = closest_topic_list_client
    
        return df
    except Exception as e: # Loop inbounded error
        print(f"[Error] topic_modeling() ran into an error! \n[Raw Error]: {e}")
        raise

def explicit_filtering(header):
    known_locs_path = "./data_prod/known_locs.json"
    with open(known_locs_path, 'r') as file:
        known_locs_dict = json.load(file)
        
    lowercase_header = header.lower()
    for key in known_locs_dict.keys():
        if (key in lowercase_header):
            return [key, known_locs_dict[key]]   
    return None

# ====== GEOLOCATION PIPELINE ======
def geolocate_articles(df):
    """
    Processes the dataaframe given by func. Does Entity Recongition/Geolocation on articles.
    
    Parameters
    ----
    df: The pandas dataframe that geolocation is being done on.

    Returns
    ---- 
    Returns a Dataframe of geolocated articles
    """
    try: 
        df = pd.concat([df['content_id'], df['Headline'], df['Body']], axis=1) # We just need the ID, Header, and Body to run the pipeline
        df["llama_prediction"] = None # Add the llama_prediction

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

        ### Explicit Mention Layer ###
        df["Explicit_Pass_1"] = df["Headline"].progress_apply(explicit_filtering)

        ### NER Pass 1 ### 
        # * This may take the longest, perhaps Truncate the output?
        df['NER_Pass_1'] = df.progress_apply(explicit_filtering_NER, axis=1) # Automatically Truncates and performs NER on first 500
        df['NER_Pass_1_Sorted'] = df['NER_Pass_1'].progress_apply(filter_loc_explicit) # We sort and pull out 'FAC' locations
        df['NER_Pass_1_Coordinates'] = df['NER_Pass_1_Sorted'].progress_apply(getLongLatsForFAC) # Using Google maps, get the longitude and latitudes

        ### Llama + NER Inference ###
        df['llama_prediction'] = df.progress_apply(predict_llama, axis=1) # This is going to flood the logs. Sorry :-(
        df['NER_prediction'] = df['llama_prediction'].progress_apply(predict_NER_def)

        df['NER_Sorted'] = df['NER_prediction'].progress_apply(format_NER) # Format
        df['NER_Sorted'] = df['NER_Sorted'].progress_apply(filter_loc) # Sort the location in priority of "FAC", "LOC", etc...

        df['NER_Sorted_Coordinates'] = df['NER_Sorted'].progress_apply(getLongLats) # Get Longitude and Latitudes
        df['Tracts'] = df.progress_apply(getTractList, axis=1) # Get the tracts

        df = df.dropna(subset=["Tracts"]) # Clean those rows that doesn't have a Tract

        return df
    except Exception as e: 
        print(f"[Fatal Error] geolocate_articles() ran into an Error! Data is not saved!\nRaw Error:{e}")
        raise Exception(f"FATAL ERROR {e}")
    return
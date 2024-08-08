import re

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from global_state import global_instance
from Model_Utils.model_Utils import explicit_filtering, process_NER, predict_llama, extractAllLocations, getAllCoordinates, getAllGeocodes, getNeighborhoods
from Model_Utils.helper_functions import clean_df

import numpy as np
from transformers import pipeline
from sklearn.metrics import adjusted_rand_score
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

# ====== GEOLOCATION PIPELINE ======
def geolocate_articles(df):
    """
    Processes the dataaframe given by func. Does Entity Recognition and Geolocation on articles.
    
    Parameters
    ----
    df: The pandas dataframe that geolocation is being done on.

    Returns
    ---- 
    Returns a Dataframe of geolocated articles
    """
    try: 
        df = clean_df(df)

        ### Explicit Mention Pass ###
        df["Explicit_Pass"] = df["Headline"].progress_apply(explicit_filtering)

        ### NER Direct Pass ### 
        # * This may take the longest, perhaps Truncate the input?
        df["NER_Pass"] = df.progress_apply(process_NER, axis=1) # Automatically Truncates and performs NER on first 500 words
                
        ### Llama + NER Inference Pass ###
        df['LLM_Pass'] = df.progress_apply(predict_llama, axis=1)
       
        # Extract Locations from Passes
        df['Locations'] = df.progress_apply(extractAllLocations, axis=1)

        # Get the Coordinates for the Locations
        df['Coordinates'] = df['Locations'].progress_apply(getAllCoordinates)

        # Geocode the Coordinates (Get the Tract and County)
        df[['Tracts', 'Counties']] = df.progress_apply(lambda row: pd.Series(getAllGeocodes(row['Locations'], row['Coordinates'])), axis=1)
        
        # Get the Neighborhoods
        df["Neighborhoods"] = df.progress_apply(getNeighborhoods, axis=1)

        # Drop the rows that are missing information
        print("[DEBUG] Data Frame ", df)
        df = df.dropna(subset=["Locations", "Coordinates", "Tracts", "Counties", "Neighborhoods"]) # Clean the rows that are missing information
        print("[DEBUG] Data Frame after dropping NaN ", df)
        
        return df
    except Exception as e: 
        print(f"[Fatal Error] geolocate_articles() ran into an Error! Data is not saved!\nRaw Error:{e}")
        raise Exception(f"FATAL ERROR {e}")
    return
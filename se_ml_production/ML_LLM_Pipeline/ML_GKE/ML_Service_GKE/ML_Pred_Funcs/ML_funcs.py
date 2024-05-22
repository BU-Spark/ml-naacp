import re
import json
import pandas as pd
from bs4 import BeautifulSoup

from tqdm import tqdm
tqdm.pandas()

from global_state import global_instance
from Model_Utils.model_Utils import explicit_filtering, explicit_filtering_NER, filter_loc_explicit, getLongLatsForFAC, predict_llama, predict_NER_def, format_NER, filter_loc, getLongLats, getTractList


# ====== TOPIC MODELING PIPELINE ======
def topic_modeling(df):
    unseen_articles = df
    try:
        openai_labels = global_instance.get_data("heir_data")
        unseen_articles = unseen_articles.dropna(subset=['content_id'])
        topics, probs = global_instance.get_data("nlp_topic").transform(unseen_articles['body']) # get bertopics for each article
        unseen_articles['bertopic_topic_label'] = topics

        # add open ai label to bglobe dataframe in new column
        unseen_label_name = [openai_labels[unseen_articles['bertopic_topic_label'][i]]['openai'] 
                      if int(unseen_articles['bertopic_topic_label'][i]) != -1 else "" for i in range(len(unseen_articles))]
        unseen_articles['openai_labels'] = unseen_label_name
        return unseen_articles
    except Exception as e: # Loop inbounded error
        print(f"[Error] topic_modeling() ran into an error! \n[Raw Error]: {e}")
        raise
    return unseen_articles

def explicit_filtering(header):
    known_locs_path = "./geodata/known_locs.json"
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
    Processes a chunk of indices from the given dataset. Does Entity Recongition/Geolocation on articles.
    
    Parameters
    ----
    df: The pandas dataframe that gelocation is being done on.

    Returns
    ---- 
    Returns a Dataframe of geolocated articles
    """
    try: 
        df = pd.concat([df['content_id'], df['hl1'], df['body']], axis=1) # We just need the ID, Header, and Body to run the pipeline
        df["llama_prediction"] = None # Add the llama_prediction

        # Remove duplicates based on Headers
        duplicates = df.duplicated(subset=['hl1']) 
        print(f"[INFO] Duplicates in DF:\n {duplicates.value_counts()}")
        df = df.drop_duplicates(subset=['hl1'])

        # Clean the HTML in the body and header -> Regex Cleaner 
        func_clean_html = lambda x: BeautifulSoup(x, "html.parser").get_text() # HTML Cleaner
        df['body'] = df['body'].progress_apply(func_clean_html)
        df['hl1'] = df['hl1'].progress_apply(func_clean_html)
        func_clean_regex = lambda x: ' '.join([item for item in re.findall(r'[A-Za-z0-9!@#$%^&*().]+', x) if len(item) > 1]) # Regex Cleaner
        df['body'] = df['body'].progress_apply(func_clean_regex)
        df['hl1'] = df['hl1'].progress_apply(func_clean_regex)

        ### Explicit Mention Layer ###
        df["Explicit_Pass_1"] = df["hl1"].progress_apply(explicit_filtering)

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
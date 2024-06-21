import pandas as pd
from tqdm import tqdm

import secret
from global_state import global_instance
from processingUtils import get_sentences, get_snippet, check_snippets, run_entity_recognition, run_pipeline


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


# ====== ENTITY RECOGNITION PIPELINE ======
def process_data(chunk, df, data_schema, data_packaging_scheme, nlp_ner):
    """
    Processes a chunk of indices from the given dataset. Does Entity Recongition.
    
    Parameters
    ----
    df: The pandas dataframe that entity recognition is being done on.
    chunk: The chunk of indices to proceses in the given df.

    Returns
    ---- 
    A list of dictionary items that constitute data for an article.
    """
    ignore_article_types = ["National News", "International News", "Programs", "Digital Mural", "Jazz", "Celtic"]
    
    discarded_articles = []
    dataset_df = data_schema
    neighborhoods = set()
    census_tracts = set()
    try: 
        for idx in tqdm(chunk, desc='Processing Entity Recognition'):
            try: # Maybe nested 'try:' is cursed
                if df['Section'][idx] not in ignore_article_types and df['Type'][idx] == 'Article':
                    headline = str(df['Label'][idx])
                    text = str(df['Body'][idx])
                    
                    sentences = get_sentences(text)
            
                    # # get lede first 5 sentences, can change the number of sentences
                    text_5 = get_snippet(sentences, 5)
                    text_10 = get_snippet(sentences, 5, False) # get sentences 5-10
                    text_remain = get_snippet(sentences, 5, False, True)
                    

                    # get entities, returns existing entities that have been seen before and new entities as sets 
                    check_order = [
                        (run_entity_recognition(headline, nlp_ner, global_instance.get_data("drop_geos"), global_instance.get_data("saved_geocodes")), "headline"), 
                        (run_entity_recognition(text_5, nlp_ner, global_instance.get_data("drop_geos"), global_instance.get_data("saved_geocodes")), "first 5 sentences"), 
                        (run_entity_recognition(text_10, nlp_ner, global_instance.get_data("drop_geos"), global_instance.get_data("saved_geocodes")), "next 5 sentences"),
                        (run_entity_recognition(text_remain, nlp_ner, global_instance.get_data("drop_geos"), global_instance.get_data("saved_geocodes")), "remaining text")
                    ]
            
                    for (entities, method) in check_order:
                        check_text, location_geocode, existing_loc_geocode = check_snippets(secret.API_KEY, entities[1], entities[0])
                        if not check_text:
                            discarded_articles.append(df['Tagging'][idx])
                            break 
        
                    # No Census tracts we want is detected
                    if (len(existing_loc_geocode) == 0 and len(location_geocode) == 0):
                        discarded_articles.append(df['Tagging'][idx])
                        continue
                    
                    pipeline_output = run_pipeline(
                        global_instance.get_data("year"), 
                        global_instance.get_data("dsource"), 
                        global_instance.get_data("dname"), 
                        global_instance.get_data("state"), 
                        existing_loc_geocode, 
                        location_geocode, 
                        global_instance.get_data("mappings")
                        )
        
                    if (pipeline_output):
                        for output in pipeline_output:
                            if ('neighborhood' in output[list(output.keys())[0]] and 'tract' in output[list(output.keys())[0]]):
                                neighborhood = output[list(output.keys())[0]]['neighborhood']
                                census_tract = output[list(output.keys())[0]]['tract']
                                neighborhoods.add(neighborhood)
                                census_tracts.add(census_tract)
                            else:
                                print("Skipped an entry!")
                                continue
                    
                    # If we have valid entity recognition | We have both some neighborhoods and census tracts
                    if (len(neighborhoods) != 0 and len(census_tracts) != 0):
                        data_packaging_scheme(
                            dataset_df, # This is the data scheme we are using
                            df['Tagging'][idx],
                            list(neighborhoods),
                            df['Section'][idx],
                            list(census_tracts),
                            df['Byline'][idx],
                            df['Body'][idx],
                            df['Tagging'][idx],
                            df['Label'][idx],
                            df['Headline'][idx],
                            df['Publish Date'][idx],
                            "GBH", # I hard coded this as we have one main client
                            df['Paths'][idx],
                            # No Open AI Labels yet
                            method,
                            existing_loc_geocode | location_geocode
                        )
                    neighborhoods.clear()
                    census_tracts.clear()
                else:
                    discarded_articles.append(df['Tagging'][idx])
            except Exception as e: # Loop inbounded error
                print(f"[Error] process_data() ran into an error! Continuing... \n[Raw Error]: {e}")
                raise Exception(f"FATAL ERROR {e}")
        ## Convert to Pandas dataframe...
        new_df = pd.DataFrame(dataset_df)
        return new_df, discarded_articles
    except Exception as e: 
        print(f"[Fatal Error] process_data() ran into an Error! Data is not saved!\nRaw Error:{e}")
        raise Exception(f"FATAL ERROR {e}")
    return
import pandas as pd
from tqdm import tqdm
from main import app # We need to import all the state variables

import secret
from processingUtils import get_sentences, get_snippet, check_snippets, run_entity_recognition, run_pipeline

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
    
    dataset_df = data_schema
    neighborhoods = set()
    census_tracts = set()
    try: 
        for idx in tqdm(chunk, desc='Processing Entity Recognition'):
            # Maybe nested 'try:' is cursed
            try:
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
                        (run_entity_recognition(headline, nlp_ner, app.state.drop_geos, app.state.saved_geocodes), "headline"), 
                        (run_entity_recognition(text_5, nlp_ner, app.state.drop_geos, app.state.saved_geocodes), "first 5 sentences"), 
                        (run_entity_recognition(text_10, nlp_ner, app.state.drop_geos, app.state.saved_geocodes), "next 5 sentences"),
                        (run_entity_recognition(text_remain, nlp_ner, app.state.drop_geos, app.state.saved_geocodes), "remaining text")
                    ]
            
                    for (entities, method) in check_order:
                        check_text, location_geocode, existing_loc_geocode = check_snippets(secret.API_KEY, entities[1], entities[0])
                        if not check_text:
                            break 
        
                    # No Census tracts we want is detected
                    if (len(existing_loc_geocode) == 0 and len(location_geocode) == 0):
                        continue
                    
                    pipeline_output = run_pipeline(
                        app.state.year, 
                        app.state.dsource, 
                        app.state.dname, 
                        app.state.state, 
                        existing_loc_geocode, 
                        location_geocode, 
                        app.state.mappings
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
            except Exception as e: # Loop inbounded error
                print(f"[Error] process_data() ran into an error! Continuing... \n[Raw Error]: {e}")
        ## Convert to Pandas dataframe...
        new_df = pd.DataFrame(dataset_df)
        return new_df
    except Exception as e: 
        print(f"[Fatal Error] process_data() ran into an Error! Data is not saved!\nRaw Error:{e}")

    return
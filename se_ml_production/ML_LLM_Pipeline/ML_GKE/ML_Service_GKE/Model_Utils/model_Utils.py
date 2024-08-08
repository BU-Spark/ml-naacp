from tqdm import tqdm
tqdm.pandas()

import requests
from global_state import global_instance

from Model_Utils.helper_functions import load_cache, save_cache, check_time

# Try finding location from title
def explicit_filtering(header):
    known_title_locs = load_cache("./geodata/known_locs.json")
    # unwanted_entities = load_cache("./geodata/unwanted_entities.json")

    
    lowercase_header = header.lower()
    for location in known_title_locs.keys():
        if (location.lower() in lowercase_header):
            return location
            # With Cache
            # if location not in unwanted_entities["FAC"]:
            #     return location
             
    return None

# Return the first valid facility found, or organization if none are found
def valid_facility(entities, firstPass):
    # unwanted_entities = load_cache("./geodata/unwanted_entities.json")

    if (firstPass): 
        for entity in entities:
            # If it's a valid facility, return it
            # if (entity.label_ == "FAC" and entity.text not in unwanted_entities["FAC"]):
            if (entity.label_ == "FAC"):
                return entity.text
        else:
            return None
    
    # Process for the LLM Prediction Pass
    else:
        first_org = None
        for entity in entities:
            # If it's a valid facility, return it
            #if (entity.label_ == "FAC" and entity.text not in unwanted_entities["FAC"]):
            if (entity.label_ == "FAC"):
                return entity.text
            
            # If it's a valid organization, save it (but don't return in case there's a facility later on)
            # if (first_org == None and entity.label_ == "ORG" and entity.text not in unwanted_entities["ORG"]):
            if (first_org == None and entity.label_ == "ORG"):
                first_org = entity.text
        else:             
            return first_org # Return regardless of whether it's None or not 

# Run NER on the body of the article and return first valid facility
def run_NER(text, firstPass=True):
    nlp = global_instance.get_data("nlp_ner")
    try:
        if (text == None or text == ""):
            return None
        
        entities = nlp(text).ents
        return valid_facility(entities, firstPass)
        
    except Exception as error:
        print(error)
        return None
    
# Process the text in chunks and return the first valid facility found
chunk_size = 100
def chunk_processing(text, chunk_size=chunk_size):
    # Split text into smaller chunks
    chunks = split_text_into_chunks(text, chunk_size)

    # Process each chunk and return if a valid facilty is found
    for chunk in chunks:
        result = run_NER(chunk)
        if result is not None:
             return result
    return None

# Split article text into chunks of specified size
def split_text_into_chunks(text, chunk_size=chunk_size):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

@check_time
def process_NER(article, truncate=True):
    """
    Process the NER on the body of the article and return the first valid facility found. If 'truncate' is true, then we get the first 500 words.
    """
    try:
        if (article['Explicit_Pass'] != None): 
            print(f"Has location from title: {article['hl1']}")
            return None
        
        else:
            if (truncate): # Truncate the text to the first 500 words
                text = " ".join(article['Body'].split(" ")[:500])
            else:
                text = article['Body']
            return chunk_processing(text)
    except Exception as error:
        print(error)
        return None

def run_llm(title, body):
    """
    Run the LLM model on the title and body of the article.
    """
    try:
        nlp_llm = global_instance.get_data("nlp_llm")
        return nlp_llm.invoke({"headline": title, "Body": body})
    except Exception as error:
        print(error)
        return None

#TODO: Comply with token limit of 2048 for Llama
# Run the LLM model on the articles that haven't been tagged with a location yet. Then run NER on the LLM prediction
@check_time
def predict_llama(article):
    try:
        # If the article does not have an explicit location or NER location, run LLM
        if (article['Explicit_Pass'] != None):
            print(f"Has location from title: {article['hl1']}")
            return None
        elif (article['NER_Pass'] != None):
            print(f"Has location from NER: {article['hl1']}")
            return None
        else:
            llama_prediction = run_llm(article['hl1'], article['body'])
            print(llama_prediction)
            return run_NER(llama_prediction, False)
    except Exception as error:
        print(error)
        return None

# Get the locations from the most specific pass for a given article
def extractLocations(article):
    for key in ['Explicit_Pass', 'NER_Pass', 'LLM_Pass']:
        location = article.get(key)
        if location is not None:
            return location
    return None

# Make a call to the Google Maps API to get the coordinates of the location
def callGoogleMapsAPI(location):
    try:
        gmaps = global_instance.get_data("googleMapsClient").client

        # Locations are limited to Massachusetts for now
        geocode_result = gmaps.geocode(f"{location}, Massachussetts", components={"administrative_area_level": "MA", "country": "US"})
        
        if (len(geocode_result) > 0):
            longitude = geocode_result[0]['geometry']['location']['lng']
            latitude = geocode_result[0]['geometry']['location']['lat']
            return longitude, latitude
        else:
            return None
    except Exception as error:
        print(error)
        return None
    
# Get the coordinates of the location
def getCoordinates(location): # Valid labels are FAC for NER_Pass; FAC and ORG for NER_Prediction
    if (location == None or len(location) == 0): return None  

    longitude, latitude = callGoogleMapsAPI(location)
    return [longitude, latitude]

    # This is with cache
    # Only get coordinates if the location is not already known
    if (location in known_locations):
        longitude, latitude = known_locations[location]["coordinates"]
    else:
        # Get coordinates and save to cache
        longitude, latitude = callGoogleMapsAPI(location)
        known_locations[location] = {"coordinates": [longitude, latitude], "tract": None, "county": None}
        save_cache(known_locations, known_locations_path)

    return [longitude, latitude]

# Get the census tract of the location
def query_census_api(location, coordinates):
    longitude, latitude = coordinates
    base_url = f'https://geocoding.geo.census.gov/geocoder/geographies/coordinates?'
    survey_ver = f'&benchmark=4&vintage=4&layers=2020 Census Blocks&format=json'
    url = f'{base_url}x={longitude}&y={latitude}{survey_ver}'

    response = requests.get(url)

    # Check if response is valid
    if (response.status_code == 200):
        results = response.json()
        try:
            tract = results['result']['geographies']['2020 Census Blocks'][0]['TRACT']
            county = results['result']['geographies']['2020 Census Blocks'][0]['COUNTY']

            return str(tract), str(county)
        except IndexError:
            print("Unable to retrieve census geography for: " + location)
        except KeyError:
            print("Location is outside of the United States: " + location)
        except Exception as error:
            print(error)
    
    print("API call failed for: " + location + " with coordinates" + str(coordinates))
    return None, None # Return this if API call failed or no tracts found


# Get the census tract and county of the location
def geocode(location, coordinates):
    if (location is None or len(location) == 0): return None, None  

    if (coordinates is None or len(coordinates) == 0): return None, None  

    # Get the tract and county from the location
    Tract, County = query_census_api(location, coordinates)
    return Tract, County

# With Cache

# def geocode(location):
#     if (location is None or len(location) == 0): return None, None  

#     # Only geocode if it's not known
#     Tract = known_locations[location]["tract"]
#     County = known_locations[location]["county"]
#     if (Tract is None or County is None):
#         # Geocode article
#         coordinates = known_locations[location]["coordinates"]
#         Tract, County = query_census_api(location, coordinates)

#         # Save to cache
#         known_locations[location]["tract"] = Tract
#         known_locations[location]["county"] = County
#         save_cache(known_locations, known_locations_path)
    
#     return Tract, County
import re
from tqdm import tqdm
tqdm.pandas()

import requests
from global_state import global_instance

from Model_Utils.helper_functions import load_cache, save_cache, check_time

from fuzzywuzzy import fuzz

# Try finding locations from title
def explicit_filtering(header):
    known_title_locs = load_cache("./geodata/known_locs.json")
    # unwanted_entities = load_cache("./geodata/unwanted_entities.json")

    locations_list = []
    lowercase_header = header.lower()
    for location in known_title_locs.keys():
        if (location.lower() in lowercase_header):
            locations_list.append(location)
            # With Cache
            # if location not in unwanted_entities["FAC"]:
            #     return location
    
    if (len(locations_list) == 0):
        return None
    else:
        return locations_list

# Return all valid facilities and organizations found
def get_valid_entities(entities):
    # unwanted_entities = load_cache("./geodata/unwanted_entities.json")
    
    valid_facilities = []
    valid_orgs = []

    for entity in entities:
        if (entity.label_ == "FAC"): # and entity.text not in unwanted_entities["FAC"]):
            valid_facilities.append(entity.text)
        
        if (entity.label_ == "ORG"): # and entity.text not in unwanted_entities["ORG"]):
            valid_orgs.append(entity.text)
    
    valid_entities = valid_facilities + valid_orgs

    if (len(valid_entities) == 0):
        return None
    else: 
        return valid_entities

# Run NER on the body of the article and return first valid facility
def run_NER(text):
    nlp = global_instance.get_data("nlp_ner")

    try:
        if (text == None or text == ""):
            return None
        
        entities = nlp(text).ents
        valid_entities = get_valid_entities(entities)
        return valid_entities
        
    except Exception as error:
        print(error)
        return None
    
# Process the text in chunks and return the first valid facility found
chunk_size = 100
def chunk_processing(text, chunk_size=chunk_size, chunk_limit=None):
    # Split text into smaller chunks
    chunks = split_text_into_chunks(text, chunk_size)

    all_entities = []

    if chunk_limit is not None:
        if len(chunks) > chunk_limit:
            chunks = chunks[:chunk_limit]

    # Process each chunk and return if a valid facilty is found
    for chunk in chunks:
        result = run_NER(chunk)
        if result is not None:
             all_entities.extend(result)
    
    if len(all_entities) == 0:
        return None
    else:
        return all_entities

# Split article text into chunks of specified size
def split_text_into_chunks(text, chunk_size=chunk_size):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

@check_time
def process_NER(article, truncate=True):
    """
    Process the NER on the body of the article and return all valid facilities and organizations found. If 'truncate' is true, then we get the first 500 words.
    """
    try:
        if (article['Explicit_Pass'] != None): 
            print(f"Has location from title: {article['Headline']}")
            return None
        
        else:
            if (truncate): # Truncate the text to the first 500 words
                chunk_limit = 5
            else:
                chunk_limit = None
            return chunk_processing(article["Body"], chunk_limit=chunk_limit)
    except Exception as error:
        print(error)
        return None

def run_llm(title, body):
    """
    Run the LLM model on the title and body of the article.
    """
    try:
        nlp_llm = global_instance.get_data("nlp_llm")
        return nlp_llm.invoke({"headline": title, "body": body})
    except Exception as error:
        print(error)
        return None

def filter_llama_output(log):
    # Define regex patterns to match the lines we want to remove
    llama_print_timings_pattern = re.compile(r'llama_print_timings:.*')
    llama_generate_pattern = re.compile(r'Llama.generate:.*')

    lines = log.split('\n')

    filtered_lines = []

    for line in lines:
        # If the line matches any of the unwanted patterns, skip it
        if llama_print_timings_pattern.match(line) or llama_generate_pattern.match(line):
            continue

        filtered_lines.append(line.strip())

    # Join the filtered lines back into a single string
    filtered_log = '\n'.join(filtered_lines)
    
    return filtered_log

# Run the LLM model on the articles that haven't been tagged with a location yet. Then run NER on the LLM prediction
@check_time
def predict_llama(article):
    try:
        # If the article does not have an explicit location or NER location, run LLM
        if (article['Explicit_Pass'] != None):
            print(f"Has location from title: {article['Headline']}")
            return None
        elif (article['NER_Pass'] != None):
            print(f"Has location from NER: {article['Headline']}")
            return None
        else:
            truncated_text = article['Body'][:4000]
            llama_prediction = run_llm(article['Headline'], truncated_text)
            cleaned_prediction = filter_llama_output(llama_prediction)
            print(f"\nLlama 3.1 Prediction: \n{cleaned_prediction} \n")
            
            valid_entities = run_NER(cleaned_prediction)
            return valid_entities
        
    except Exception as error:
        print(error)
        return None

# Helper functions for geolocation
def normalize_location(location):
    location = location.lower().strip()
    if location.startswith("the "):
        location = location[4:]
    return location

def are_same_location(loc1, loc2, threshold=85):
    norm_loc1 = normalize_location(loc1)
    norm_loc2 = normalize_location(loc2)
    similarity = fuzz.token_set_ratio(norm_loc1, norm_loc2)
    return similarity >= threshold

def get_unique_locations(locations):
    unique_locations = []
    for loc in locations:
        if not any(are_same_location(loc, unique_loc) for unique_loc in unique_locations):
            unique_locations.append(loc)
    return unique_locations

# Get all locations from the article
def extractAllLocations(article):
    locations_list = []
    for key in ['Explicit_Pass', 'NER_Pass', 'LLM_Pass']:
        location = article.get(key)
        if location is not None:
            locations_list.extend(location)
            break
    
    if len(locations_list) == 0:
        return None
    else:
        # Ensure unique locations considering variations
        unique_locations = get_unique_locations(locations_list)
        return unique_locations

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
def getCoordinates(location): 
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

# Get all the coordinates of the locations
def getAllCoordinates(locations):
    coordinates_list = []

    if (locations == None): return None
    
    for location in locations:
        coordinates = getCoordinates(location)
        if coordinates is not None:
            coordinates_list.append(coordinates)
    return coordinates_list

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

def getAllGeocodes(locations, coordinates):
    tracts = []
    counties = []

    if (locations == None): return None, None
    if (coordinates == None): return None, None

    for i, location in enumerate(locations):
        Tract, County = geocode(location, coordinates[i])
        tracts.append(Tract)
        counties.append(County)

    return tracts, counties
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

def getNeighborhoods(articles):
    tracts = articles['Tracts']

    neighborhoods = []
    neighborhood_map = global_instance.get_data("neighborhoods")

    if (tracts == None): return None

    for i, tract in enumerate(tracts):
        if (tract == None): continue
        
        neighborhood = neighborhood_map.get(tract)
        if neighborhood is not None:
            neighborhoods.append(neighborhood)

    return neighborhoods
    # With Cache 
    #         if (neighborhood[0] == "Unknown Neighborhood"):
    #             location = articles['Locations'][i]
    #             if (tract not in unknown_tracts):
    #                 unknown_tracts[tract] = {"County": articles['Counties'][i], "Locations": [location]}
    #                 print(f"Unknown neighborhood for location: {location} with tract: {tract}")
    #             elif (location not in unknown_tracts[tract]["Locations"]):
    #                 unknown_tracts[tract]["Locations"].append(location)
    #                 print(f"Unknown neighborhood for location: {location} with tract: {tract}")
    #         neighborhoods.extend(neighborhood)

    # save_cache_to_file(unknown_tracts, unknown_tracts_path)
    # return neighborhoods
from tqdm import tqdm
tqdm.pandas()

from global_state import global_instance

from Model_Utils.helper_functions import load_cache, save_cache, check_time
from Model_Utils.location_Utils import get_valid_title_locations, get_valid_entities
from Model_Utils.geocoding_Utils import geocode

# Try finding locations from title
def explicit_filtering(header):
    valid_locations = get_valid_title_locations(header.lower())
    return valid_locations

# Run NER on the body of the article and return first valid facility
@check_time
def run_NER(text, truncate=True):

    if (truncate): # Truncate the text to the first 500 words
        text = ' '.join(text.split()[:500])
    
    if (text == None or text == ""):
        return None
    
    nlp = global_instance.get_data("nlp_ner")
    try:  
        entities = nlp(text).ents
        valid_entities = get_valid_entities(entities)
        return valid_entities
        
    except Exception as error:
        print("Failed running the NER: ", error)
        return None

def process_NER(article, truncate=True):
    """
    Process the NER on the body of the article and return all valid facilities and organizations found. If 'truncate' is true, then we get the first 500 words.
    """
    if (article['Explicit_Pass'] != None): 
        print(f"Has location from title: {article['Headline']}")
        return None
    else:
        valid_entities = run_NER(article['Body'], truncate)
        return valid_entities

# Run the LLM model on the title and body of the article.
@check_time
def run_llm(title, body):
    nlp_llm = global_instance.get_data("nlp_llm")

    try:
        llama_prediction = nlp_llm.invoke({"headline": title, "body": body})
        return llama_prediction
    except Exception as error:
        print("Failed running the LLM: ", error)
        return None

def process_LLM(article, truncate=True):
    """
    Try to predict the location of the article using the LLM model. Then run NER on prediction to obtain locations.
    """

    # If the article does not have an explicit location or NER location, run LLM
    if (article['Explicit_Pass'] != None):
        print(f"Has location from title: {article['Headline']}")
        return None
    elif (article['NER_Pass'] != None):
        print(f"Has location from NER: {article['Headline']}")
        return None
    
    else:
        llama_prediction = run_llm(article['Headline'], article['Body'])
        valid_entities = run_NER(llama_prediction, truncate)
        return valid_entities
        
# Get all locations from the article
def getAllLocations(article):
    locations_list = []
    for key in ['Explicit_Pass', 'NER_Pass', 'LLM_Pass']:
        location = article.get(key)
        if location is not None:
            locations_list.extend(location)
            break
    
    if len(locations_list) == 0:
        return None
    else:
        return locations_list
    
# Get the coordinates of the location
def getCoordinates(location): 
    if (location == None or len(location) == 0): return None  

    gmaps = global_instance.get_data("googleMapsClient")
    # Only on Massachussets for now
    coordinates = gmaps.getCoordinates(location, state="Massachussetts", components={"administrative_area_level": "MA", "country": "US"})

    return coordinates # [longitude, latitude]

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

# Get the geocodes (tract and county) of each of the locations
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

def getAllNeighborhoods(articles):
    tracts = articles['Tracts']

    neighborhoods = []
    neighborhood_map = global_instance.get_data("neighborhoods")

    if (tracts == None): return None

    for i, tract in enumerate(tracts):
        if (tract == None): continue

        neighborhood = neighborhood_map.get(tract)
        if neighborhood is not None:
            neighborhoods.append(neighborhood)
        else:
            neighborhoods.append("Unknown Neighborhood")

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
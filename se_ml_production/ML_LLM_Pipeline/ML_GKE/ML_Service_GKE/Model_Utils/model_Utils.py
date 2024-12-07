from tqdm import tqdm
tqdm.pandas()

from global_state import global_instance

from Model_Utils.helper_functions import load_cache, save_cache, check_time
from Model_Utils.location_Utils import get_title_entities, get_valid_entities, get_main_5
from Model_Utils.geocoding_Utils import geocode
from Mongo_Utils.mongo_neighborhoods import create_neighborhood

# Try finding locations from title
def explicit_filtering(header):
    all_locations = get_title_entities(header.lower())
    return all_locations

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
        all_locations = get_valid_entities(entities)
        return all_locations
        
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
        all_locations = run_NER(article['Body'], truncate)
        return all_locations

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
        text = article["Body"]
        if truncate:
            text = ' '.join(text.split()[:500])
        llama_prediction = run_llm(article['Headline'], text)
        all_locations = run_NER(llama_prediction, truncate)
        return all_locations

# Get all locations from the article
def getAllLocations(article):
    locations_dict = {}
    for key in ['Explicit_Pass', 'NER_Pass', 'LLM_Pass']:
        location = article[key]
        if location is not None:
            locations_dict = location
            break
    
    if len(locations_dict) == 0:
        return None
    elif len(locations_dict["FAC"]) == 0 and len(locations_dict["ORG"]) == 0:
        return None
    else:
        return locations_dict
    
# Get the coordinates of the location
def getCoordinates(location): 
    gmaps = global_instance.get_data("googleMapsClient")
    known_locations_path = "./data_prod/known_locations.json"
    known_locations = load_cache(known_locations_path)

    try:
        if (location == None or len(location) == 0): return None  
        # Only get coordinates if the location is not already known
        if (location in known_locations):
            longitude, latitude = known_locations[location]["coordinates"]
        else:
            # Get coordinates and save to cache
            longitude, latitude, city = gmaps.callGoogleMapsAPI(location)
            if (longitude is None or latitude is None): return None
            
            known_locations[location] = {"coordinates": [longitude, latitude], "city": city, "state": None, "tract": None, "county": None}
            save_cache(known_locations, known_locations_path)

        return [longitude, latitude]
    except Exception as error:
        print(f"[ERROR] Error getting coordinates for {location}: {error}")
        return None

def getAllCoordinates(locations, all_locations):
    if (locations == None or len(locations) == 0): return None, all_locations

    try: 
        found_locations = {"FAC": {}, "ORG": {}}

        for type in ["FAC", "ORG"]:
            if locations[type] is None or len(locations[type]) == 0: continue
            
            unique_locations = list(set(locations[type]))
            for location in unique_locations:
                coordinates = getCoordinates(location)
                if coordinates is not None:
                    found_locations[type][location] = coordinates
        
        if len(found_locations["FAC"]) == 0 and len(found_locations["ORG"]) == 0:
            return None, locations
        else:
            return found_locations, locations
    except Exception as error:
        print(f"[ERROR] Error processing locations: {locations}, Error: {error}")
        return None

# Get the main locations and their coordinates
def getMainLocations(article):
    all_locations = article["all_locations"]
    valid_locations = article["locations"]
    if all_locations is None or valid_locations is None: return None, None 

    full_valid_locations = {"FAC": [], "ORG": []}
    for type in ["FAC", "ORG"]:
        for location in all_locations[type]:
            if location in valid_locations[type].keys():
                full_valid_locations[type].append(location)

    main_locations = get_main_5(full_valid_locations["FAC"], full_valid_locations["ORG"])
    if main_locations is None: return None, None
    
    main_coords = []
    for location in main_locations:
        main_coords.append(valid_locations["FAC"].get(location, valid_locations["ORG"].get(location)))
    return main_locations, main_coords

# Get the census tract and geocode info of the location
def getAllGeocodes(locations, coordinates):
    tracts = []
    counties = []
    states = []
    cities = []

    if (locations == None or len(locations) == 0): return None, None, None, None
    if (coordinates == None or len(coordinates) == 0): return None, None, None, None
    
    for i, location in enumerate(locations):
        try:
            Tract, County, State, City = geocode(location, coordinates[i])
        except Exception as error:
            print(f"[ERROR] Error geocoding location: {location}, Error: {error}")
            Tract, County, State, City = None, None, None, None
            
        tracts.append(Tract)
        counties.append(County)
        states.append(State)
        cities.append(City)

    return tracts, counties, states, cities

# Get the neighborhoods of each of the locations
def getAllNeighborhoods(articles):
    tracts = articles['tracts']

    neighborhoods = []
    neigh_map = global_instance.get_data("neigh_map")

    if (articles['locations'] == None or len(articles['locations']) == 0): return None
    if (articles['coordinates'] == None or len(articles['coordinates']) == 0): return None
    if (tracts == None or len(tracts) == 0): return None
 
    for i, tract in enumerate(tracts):
        if (tract == None): 
            neighborhoods.append("No Neighborhood")
            continue

        neighborhood = neigh_map.get(tract)
        if neighborhood is not None:
            neighborhoods.append(neighborhood)
        else:
            city = articles['cities'][i]
            if city is None:
                neighborhoods.append("Unknown Neighborhood")
            else:
                create_neighborhood(tract, city)
                neighborhoods.append(city)
    return neighborhoods

# def getAllNeighborhoods(articles):
#     tracts = articles['tracts']

#     neighborhoods = []

#     if (articles['locations'] == None or len(articles['locations']) == 0): return None
#     if (articles['coordinates'] == None or len(articles['coordinates']) == 0): return None
#     if (tracts == None or len(tracts) == 0): return None

#     for i, tract in enumerate(tracts):
#         if (tract == None): 
#             neighborhoods.append("No Neighborhood")
#             continue

#         neighborhood = tract_map.get(tract)
#         if neighborhood is not None:
#             neighborhoods.append(neighborhood)
#         else:
#             city = articles['cities'][i]
#             if city is None:
#                 neighborhoods.append("Unknown Neighborhood")
#             else:
#                 create_neighborhood(tract, city)
#                 neighborhoods.append(city)

#     return neighborhoods

def removeRepeatedCoords(row):
    coordinates = row['coordinates']
    if coordinates is None or not isinstance(coordinates, list):
        return row
    
    coord_counts = {}
    indexes_to_remove = set()
    
    for i, coord in enumerate(coordinates):
        if coord is None or coord[0] is None or coord[1] is None:
            print(f"[WARNING] Removing unprocessed location: {row['locations'][i]}")
            indexes_to_remove.add(i)
            continue
        elif row['tracts'][i] is None:
            print(f"[WARNING] Removing location with no tract: {row['locations'][i]}")
            indexes_to_remove.add(i)
            continue
        
        coord_tuple = tuple(coord) 
        if coord_tuple in coord_counts:
            indexes_to_remove.add(i)
        else:
            coord_counts[coord_tuple] = i

    # Remove duplicates from each column
    for column in ['locations', 'coordinates', 'tracts', 'counties', 'states', 'cities', 'neighborhoods']:
        if isinstance(row[column], list):
            row[column] = [v for i, v in enumerate(row[column]) if i not in indexes_to_remove]
    

    return row

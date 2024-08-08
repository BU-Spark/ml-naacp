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
    for key in ['Explicit_Pass', 'NER_Pass', 'LLM_2_Pass', 'LLM_3_1_Pass']:
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
neigh_tract_dict = {
	"Fenway" : ["010103", "010104", "010204", "010408", "010404", "010403", "981501", "010405", "010206", "010205"],
	"Downtown": ["030302", "070202", "070102", "030301", "070104", "070103", "070201"],
	"Beacon Hill": ["020200", "020302", "020101", "981700"],
	"Dorchester" : [
	"092400", "091400", "090300", "091800", "092300", "100601", "090901", 
	"100400", "090100", "091001","090200", "100200", "091700", "092200", "090700",
	"091500", "091300", "100300", "100100", "092000", "100500", "100800", "100603",
	"091200", "100700", "092101", "091900", "091600", "091100"
	],
	"Mattapan": ["100900", "101002", "101102", "981100", "101001","101101"],
	"Jamaica Plain": [
	"120103", "981800", "110105", "120600", "120700", "120301", "081200", "120105","081101",
	"981000", "120500", "120104", "120201", "110106", "081301", "120400"
	],
	"Roslindale": ["110502", "110104", "110501", "110401", "140106", "110301", "110607", "110403","110201"],
	"Roxbury": [
	"081500", "080500", "070801", "080100", "081800", "980300", "082000", "080601", "081700", "080300",
	"090600", "081400", "090400", "070901", "082100", "081900", "081302","080401"
	],
	"West End": ["020304", "020301", "020305"],
	"Longwood": ["010300", "081001"],
	"South Boston": ["061101", "060700", "060101", "061201", "061000", "060800", "981201", "060200", "061202", "060400", "061203", "060301", "060601", "060501"],
	"Back Bay": ["010702", "010701", "010802", "010801", "010500", "010600"],
	"Charlestown": ["040100", "040300", "040401", "040600", "040801", "040200"],
	"Allston": ["000604", "000804", "000703", "000704", "000806", "000101", "000807", "000701", "000805"],
	"Hyde Park": ["140107", "140201", "140105", "980700", "140300", "140202", "140400", "140102"],
	"East Boston": ["050500", "050600", "981502", "050101", "981300", "050901", "050300", "050700", "050400", "051000", "981600", "051200", "050200", "051101"],
	"South End": ["070301", "070302", "070502", "070501", "071101", "070600", "070700", "070902", "070802", "071201", "070402"],
	"West Roxbury": ["980900", "130406", "981900", "130404", "110601", "130300", "130402", "130200", "130101"],
	"South Boston Waterfront": ["981202", "060602", "060603", "061204", "060604"],
	"North End": ["030200", "030100", "030500", "030400"],
	"Cambridge": ["354300", "354200", "353102", "353600", "352300", "354100", "359400", "353300", "353700", "353200", 
	"354601", "355000", "354602", "354000", "354901", "354902", "353900", "354700", "352102", "354500", "354800", "352600", 
	"354400", "353101", "352900", "353000", "352101", "353800", "352500", "352400", "352700", "352200", "352800", 
    "365100", "361300"
  	],
	"Chelsea": ["160400", "160103", "160102", "160300", "160601", "160602", "160501", "160502", "160200"],
	

}

def string_to_list(s):
    if(s != ''):
        return [s]
    else:
        return []  # Return the string as a single-element list

def find_neighborhood_by_tract(search_dict, tract_to_find):
	for key, values in search_dict.items():
		if (tract_to_find in values):
			return key
	return "Unknown Neighborhood"

def locateNeighborhoods(tract):
	if (tract == None):
		return None

	query = find_neighborhood_by_tract(neigh_tract_dict, tract)
	if (query != None):
		return string_to_list(query)
	else: 
		return string_to_list("None")

def getNeighborhoods(articles):
    tracts = articles['Tracts']

    neighborhoods = []

    if (tracts == None): return None

    for i, tract in enumerate(tracts):
        neighborhood = locateNeighborhoods(tract)
        if neighborhood is not None:
            neighborhoods.extend(neighborhood)

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
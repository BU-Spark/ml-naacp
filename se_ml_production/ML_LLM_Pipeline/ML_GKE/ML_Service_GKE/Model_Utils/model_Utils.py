import ast
import requests
from global_state import global_instance

def explicit_filtering(header):
    known_locs_path = "./geodata/known_locs.json"
    with open(known_locs_path, 'r') as file:
        known_locs_dict = json.load(file)
        
    lowercase_header = header.lower()
    for key in known_locs_dict.keys():
        if (key in lowercase_header):
            return [key, known_locs_dict[key]]   
    return None

predict_NER = lambda x: [(entity, entity.label_) for entity in global_instance.get_data("nlp_ner")(x).ents] if (x != None and x != "") else None
def predict_NER_def(x):
    """
    Run NER on a Body of text.
    """
    try:
        return predict_NER(x)
    except Exception as e:
        print(e)
        return None
    
def explicit_filtering_NER(col, truncate=True):
    """
    Wrapper for NER on for the first pass of NER on Body. If 'truncate' is true, then we get the first 500 words.
    """
    try:
        if (col['Explicit_Pass_1'] != None): # We already found an explicit mention in the title
            print(f"Passed on {col['Headline']}")
            return None
        else:
            if (truncate):
                return predict_NER_def(" ".join(col['Body'].split(" ")[:500]))  
            else:
                return predict_NER_def(col['Body'])
    except Exception as e:
        print(e)
        return None
    
def filter_loc_explicit(x):
    """
    Filter the explicit mentions in first NER Pass
    """
    if (x == None):
        return None
    res = []
    for tup in x:
        if (len(tup) >= 2): 
            if (("GPE" in tup[1] and "Boston" not in tup[0] and "Massachusetts" not in tup[0])
                or ("ORG" in tup[1]) 
                or ("FAC" in tup[1])
                or ("LOC" in tup[1])
            ):
                res.append((tup[0], tup[1].strip()))
    priority = {'FAC': 1, 'ORG': 2, 'LOC': 3, 'GPE': 4}
    sorted_list = sorted(res, key=lambda x: priority[x[1]])
    
    return sorted_list

def getLongLatsForFAC(x):
    """
    Gets the longitude and latitude for 'FAC' locations.
    """
    if (x == None or len(x) == 0):
        return None  
    
    location = x[0][0] # (Location, Label)
    if (x[0][1] == "FAC" or "Boston" in location): # Check if we have Boston + 'FAC' Label
        response = global_instance.get_data("googleMapsClient").client.geocode(f"{location}, Boston")
    elif(x[0][1] == "FAC"): # Check if we have 'FAC' Label
        response = global_instance.get_data("googleMapsClient").client.geocode(f"{location}, Massachusetts")
    else:
        return None # Doesn't have 'FAC' Label
        
    if (len(response) == 0):
        return None  
    latitude = response[0]['geometry']['location']['lat']
    longitude = response[0]['geometry']['location']['lng']
    
    return [longitude, latitude]

def predict_llama(col):
    """
    Runs the input through an LLM prompt on a Body of text.
    """
    try:
        if (col['Explicit_Pass_1'] != None or col['NER_Pass_1_Coordinates'] != None): # We already found an explicit mention in the previous passes
            print(f"Passed on {col['Headline']}")
            return None
        else:
            return global_instance.get_data("nlp_llm").invoke({"headline": col['Headline'], "Body": col['Body']})
    except Exception as e:
        print(e)
        return None
    
def remove_first_comma(x):
    if (x[:1] == ","):
        return x[2:]
    else:
        return x

def format_NER(x):
    x = str(x)
    res = []
    if (x != None):
        input = x.replace("(","").replace("[","").replace("]","").replace("'","").split(")") 
        for word in input:
            res.append(remove_first_comma(word).strip())
    return res 

def filter_loc(x):
    res = []
    for tup in x:
        cleaned_tup = tup.strip().split(",")
        if (len(cleaned_tup) >= 2): 
            if (("GPE" in cleaned_tup[1] and "Boston" not in cleaned_tup[0] and "Massachusetts" not in cleaned_tup[0])
                or ("ORG" in cleaned_tup[1]) 
                or ("FAC" in cleaned_tup[1])
                or ("LOC" in cleaned_tup[1])
            ):
                res.append((cleaned_tup[0], cleaned_tup[1].strip()))
    priority = {'FAC': 1, 'ORG': 2, 'LOC': 3, 'GPE': 4}
    sorted_list = sorted(res, key=lambda x: priority[x[1]])
    
    return sorted_list
    
def getLongLats(x):
    """
    Get the longitude and latitudes based on NER outputs.
    """
    if (len(x) == 0):
        return None  
        
    location = x[0][0] # (Location, Label)
    if (x[0][1] == "ORG" or x[0][1] == "FAC" or "Boston" in location):
        response = global_instance.get_data("googleMapsClient").client.geocode(f"{location}, Boston")
    else:
        response = global_instance.get_data("googleMapsClient").client.geocode(f"{location}, Massachusetts")
    if (len(response) == 0):
        return None  
    latitude = response[0]['geometry']['location']['lat']
    longitude = response[0]['geometry']['location']['lng']
    return [longitude, latitude]

def query_census_api(longitude, latitude):
    """
    Queries the U.S Census API for tracts.
    """
    county = "NO COUNTY"
    url = f"https://geocoding.geo.census.gov/geocoder/geographies/coordinates?x={longitude}&y={latitude}&benchmark=Public_AR_Current&vintage=Census2020_Current&format=json"
    response = requests.get(url)
    if (response.status_code == 200):
        results = response.json()
        census_tracts = results['result']['geographies'].get('Census Tracts', [])
        if (census_tracts):
            return census_tracts[0].get('TRACT', 'No TRACT found'), county # Returning the TRACT of the first census tract found
    return "No TRACT found", county  # Return this if API call failed or no tracts found

def getTractList(col):
    """
    Get the Tracts based on Coordinate outputs.
    """
    coordinates = []
    tract_list = [] # Initialize an empty list to store TRACT information  
    
    if (col['Explicit_Pass_1'] != None): # If we got the locations from the first explicit pass
        coordinates = col['Explicit_Pass_1'][1]
    elif(col['NER_Pass_1_Coordinates'] != None): # If we got locations from the very first NER pass (specific locs only)
        coordinates = col['NER_Pass_1_Coordinates']
    elif (col['NER_Sorted_Coordinates'] != None): # Finally, if we got locations from llama + NER pass
        coordinates = col['NER_Sorted_Coordinates']
    else: # Must be a very hard/bad article :-(
        return None 
        
    longitude = coordinates[0]
    latitude = coordinates[1]
    TRACT, COUNTY = query_census_api(longitude,latitude)
    tract_list.append(TRACT)
    
    return tract_list
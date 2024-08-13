import re
from fuzzywuzzy import fuzz
from collections import Counter

from Model_Utils.helper_functions import load_cache

# Normalize a location string
def normalize_location(location):
    location = location.lower().strip() 
    
    # Remove leading terms
    leading_terms = ["a", "an", "the"]

    for term in leading_terms:
        if location.startswith(term + " "):
            location = location[len(term) + 1:]

    # Remove punctuation
    location = re.sub(r'[^\w\s]', '', location)
    
    # Expand common abbreviations
    abbreviation_map = {
        "us": "united states",
        'co': 'company',
        'pd': 'police department',
        'wgbh': 'gbh',
        'cfa': 'the harvard-smithsonian center for astrophysics',
        'pbs': 'public broadcasting service',
        'ap': 'associated press',
        'aim': 'aim (alternative investment market)',
        '&': 'and',
        'npr': 'national public radio',
        'doj': 'the justice department',
        'cdc': 'the centers for disease control and prevention',
        'ssa': 'social security administration',
        'doe': 'the energy department',
        'cbpp': 'the center on budget and policy priorities',
        'mbta': 'massachusetts bay transportation authority',
        't': 'massachusetts bay transportation authority',
        'globe': 'boston globe',
        'senate': 'capitol',
        'congress': 'capitol',
        'legislature': 'capitol',
        'justice': 'department of justice',
        'house': 'u.s. house of representatives',
        'oval office': 'the white house',
        'dese': 'department of elementary and secondary education',
        'bps': 'boston public schools',
        'dhs': 'the department of homeland security',
        'fed': 'federal reserve',
        'fbi': 'the federal bureau of investigation',
        'epa': 'the environmental protection agency',
        'cdc': 'the centers for disease control and prevention',
        'nar': 'national association of realtors',
        'adl': 'anti-defamation league',
        'cbp': 'customs and border protection',
        'cia': 'central intelligence agency',
        'fda': 'food and drug administration',
        'dep': 'department of environmental protection',
        'un': 'united nations',
        'faa': 'federal aviation administration',
        'ntsb': 'national transportation safety board',
        'dua': 'department of unemployment assistance',
        'necn': 'new england cable news',
        'nar': 'national association of realtors',
        'who': 'world health organization',
        'irap': 'international refugee assistance project',
        'ncaa': 'national collegiate athletic association',
        'council': 'city council',
        'dph': 'the department of public health',
        'usda': 'the us department of agriculture',
        'bpr': 'boston public radio',
        'blm': 'black lives matter',
        'irs': 'internal revenue service',
        'necn': 'new england cable news',
        'wh': 'white house',
        'gop': 'the republican party',
        'ps': 'public schools',
        'ma dese': 'the massachusetts department of elementary and secondary education',
    }
    for abbr, full in abbreviation_map.items():
        location = re.sub(r'\b' + abbr + r'\b', full, location)
    
    # TODO: (maybe) Handle synonyms, variants
    # TODO: (maybe) Handle country/state/city abbreviations   
    
    # Remove extra spaces
    location = re.sub(r'\s+', ' ', location)
    
    return location

# Check if two locations are the same
def are_same_location(loc1, loc2, threshold=80):
    if loc1 == loc2:
        return True
    
    if loc1 in loc2 or loc2 in loc1:
        return True
    
    similarity = fuzz.token_set_ratio(loc1, loc2)
    return similarity >= threshold

# Combine two locations if they are the same
def combine_locations(location, locations):
    if (locations is None or len(locations) == 0):
        return [location]

    # check for news subdomains
    if "gbh" in location:
        location = "gbh"
    elif "npr" in location:
        location = "npr"

    new_locs = []

    for loc in locations:
        if are_same_location(location, loc):
            # Replace the existing location if the new one is longer
            if len(location) > len(loc):
                new_locs.append(location)
            else:
                new_locs.append(loc)
        else:
            new_locs.append(loc)

    new_locs.append(location)

    return new_locs

# Check if a location can be added to the list of locations
def can_add_location(location):
    unwanted_entities = load_cache("./data_prod/unwanted_locations.json")

    if location in unwanted_entities:
        return False
    elif invalid_location(location):
        return False
    else:
        return True
    
# Check if a location is unwanted with regex
def invalid_location(location):
    # List of common unwanted entity types
    unwanted_places = [
        r'\bstreet\b', 
        r'\bsquare\b', 
        r'\bavenue\b', 
        r'\bboulevard\b',
        r'\broad\b', 
        r'\blane\b', 
        r'\bdrive\b', 
        r'\bdriveway\b',
        r'\bhighway\b',
        r'\bfreeway\b'
    ]
    
    # Create a combined regex pattern
    pattern = re.compile('|'.join(unwanted_places))
    
    # Check if the location matches any unwanted entity type
    if pattern.search(location):
        return True
    return False

# Try finding locations from title
def get_valid_title_locations(header):
    known_title_locs = load_cache("./data_prod/known_locs.json")
    known_locations = known_title_locs.keys()

    locations_list = []
    for location in known_locations:
        loc = normalize_location(location)
        if (loc in header and can_add_location(loc)):
            locations_list.append(loc)
        
        if (len(locations_list) == 5):
            break
    
    if (len(locations_list) == 0):
        return None
    else:
        return locations_list

# Get the top 5 most common locations
def get_main_5(facilities, organizations):

    fac_freq = Counter(facilities)
    org_freq = Counter(organizations)

    top_fac = fac_freq.most_common(1) if facilities else []
    top_org = org_freq.most_common(1) if organizations else []

    combined = facilities + organizations
    combined_freq = Counter(combined)

    if top_fac:
        combined_freq.pop(top_fac[0][0], None)
    if top_org:
        combined_freq.pop(top_org[0][0], None)
    
    top_combined = combined_freq.most_common(3)

    top_entities = top_fac + top_org + top_combined
    
    top_entities = [entity[0] for entity in top_entities]

    return top_entities


def add_entity(entity, valid_list):
    loc = normalize_location(entity)
    if (can_add_location(loc)):
        valid_list = combine_locations(loc, valid_list)
    
    return valid_list

# Return all valid facilities and organizations found
def get_valid_entities(entities):
    valid_facs = []
    valid_orgs = []

    if (entities is None or len(entities) == 0):
        return None

    for entity in entities:
        if (entity.label_ == "FAC"):
            valid_facs = add_entity(entity.text, valid_facs)
        elif (entity.label_ == "ORG"):
            valid_orgs = add_entity(entity.text, valid_orgs)

    valid_entities = get_main_5(valid_facs, valid_orgs)
    
    if (len(valid_entities) == 0):
        return None
    else:
        return valid_entities
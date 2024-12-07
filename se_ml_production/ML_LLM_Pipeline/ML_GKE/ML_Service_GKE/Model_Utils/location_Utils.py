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
# If a location is in the title, use that as the article's location
def get_title_entities(header):
    known_title_locs_path = "./data_prod/known_locations.json"
    known_title_locs = load_cache(known_title_locs_path)
    known_title_locations = known_title_locs.keys()
    # Look through the header for known locations
    locations_list = []
    for location in known_title_locations:
        loc = normalize_location(location)
        if (loc in header and can_add_location(loc)):
            locations_list.append(loc)
        
    
    if (len(locations_list) == 0):
        return None
    else:
        all_locations = {"FAC": locations_list, "ORG": []}

        return all_locations

# Get the top 5 most common locations
def get_main_5(facilities, organizations):
    fac_freq = Counter(facilities)
    org_freq = Counter(organizations)
    
    top_entities = [] + fac_freq.most_common(1) + org_freq.most_common(1)

    top_entities.extend([entity for entity in fac_freq.most_common() if entity not in top_entities])

    top_entities.extend([entity for entity in org_freq.most_common(5 - len(top_entities)) if entity not in top_entities])

    if len(top_entities) == 0: return None
    
    top_entities = [entity[0] for entity in top_entities]
    return top_entities

# Add a location to the list of valid locations
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

    if (len(valid_facs) == 0 and len(valid_orgs) == 0):
        return None
    else:
        all_locations = {"FAC": valid_facs, "ORG": valid_orgs}

        return all_locations
    
# Get the best location name from a list of locations
def best_location(locations):
	if len(locations) == 1: return locations[0]
	
	# Remove some/(most?) abbreviations
	locations = [location for location in locations if len(location) > 3]
	if len(locations) == 1: return locations[0]

	sorted_locations = sorted(locations, key=len)

	def find_substring(sorted_list):
		for i in range(len(sorted_list)):
			substring = sorted_list[i]
			
			# Skip single words
			if len(substring.split(" ")) == 1: continue
			
			# Check if this substring is in at least some of the others
			# Since it's ordered, getting the first one that has a match might give us the 'best' location name
			count = sum(1 for other in sorted_list if substring in other and other != substring)
			if count > 0: return substring
			
		return None

	# Get the result
	result = find_substring(sorted_locations)
	if result: return result
	
	if len(sorted_locations[0].split(" ")) > 1:
		return sorted_locations[0]
	else:
		return sorted_locations[1]
    

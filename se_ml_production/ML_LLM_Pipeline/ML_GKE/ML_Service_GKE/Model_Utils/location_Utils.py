import re
from fuzzywuzzy import fuzz

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
        'st': 'street',
        'ave': 'avenue',
        'blvd': 'boulevard',
        'rd': 'road',
        'GBH': 'WGBH, Boston Public Media',
        'CfA': 'the Harvard-Smithsonian Center for Astrophysics',
        'Granite Broadcasting': 'Granite Broadcasting Holdings',
        'Mystic Aquarium': 'the Mystic Aquarium',
        'North Shore Aquarium': 'North Shore N.E. Aquarium',
        'Blurb': 'Blurb.com',
        'Sweetwater': 'Sweetwater Co.',
        'Beverly PD': 'Beverly Police',
        'PBS': 'Public Broadcasting Service',
        'ITV': 'ITV plc',
        'ITV Global': 'ITV Global Entertainment Ltd',
        'GBH Channel 2': 'GBH 2',
        'BBC 2': 'BBC Two',
        'Parabola': 'the Parabola Center',
        'Treez': 'Treez of Lyfe',
        'NOVA PBS': 'NOVA',
        'Shutterstock Images': 'Shutterstock',
        'Eky Studio': 'bezikus Eky Studio',
        'AP': 'Associated Press',
        'Newton Marriott Hotel': 'Newton Marriott',
        'AIM': 'AIM (Alternative Investment Market)',
        'MassReconnect Program': 'MassReconnect',
        'McKinsey & Company': 'McKinsey Company',
        'NPR': 'National Public Radio',
        'Purdue': 'Purdue Pharma',
        'Johnson & Johnson': 'Johnson Johnson McKesson',
        'McKesson': 'Johnson Johnson McKesson',
        'Wal-Mart': 'Walmart',
        'DOJ': 'the Justice Department',
        'CDC': 'the Centers for Disease Control and Prevention',
        'GBH Studios': 'GBH Studio',
        'Boston Library': 'the Boston Public Library',
        'NEC Quartet': 'the New England Conservatory Fellowship String Quartet',
        'SSA': 'Social Security Administration',
        'DOE': 'the Energy Department',
        'CBPP': 'the Center on Budget and Policy Priorities',
        'T': 'MBTA',
        'Orange Line MBTA': 'the Orange Line',
        'GBH Newsroom': 'GBH News',
        'Northeastern': 'Northeastern University',
        'Red Line MBTA': 'the Red Line',
        'Fraser Studio': 'GBH Fraser Performance Studio',
        'GBH Music Studio': 'GBH Music',
        'Rasa Quartet': 'Unique Music Adventure Rasa Quartet',
        'Baroque': 'Boston Baroque',
        'Rasa Quartet': 'The Rasa String Quartet',
        'Grosso': 'Concerto Grosso',
        'Harmonia': 'Harmonia Artificioso',
        'Globe': 'Boston Globe',
        'Senate': 'U.S. Senate',
        'AP': 'Associated Press',
        'Nevada Independent': 'the Nevada Independent',
        'DESE': 'Department of Elementary and Secondary Education',
        'BPS': 'Boston Public Schools',
        'Boston Schools': 'the Boston Public Schools',
        'DHS': 'the Department of Homeland Security',
        'McDonald\'s': 'McDonald',
        'Council': 'City Council',
        'K12 Security': 'K12 Security Information Exchange',
        'Senate Homeland Security Committee': 'the U.S. Senate Committee on Homeland Security and Governmental Affairs',
        'Worcester Council': 'Worcester City Council',
        'Nubian': 'Nubian Square',
        'Copley': 'Copley Square',
        'Sunrise Movement': 'the Sunrise Movement Socialist Alternative',
        'Democratic Socialists': 'the Democratic Socialist party',
        'DPH': 'the Department of Public Health',
        'COVID-19': 'COVID',
        'Boston Hall': 'Boston City Hall',
        'BPR': 'Boston Public Radio',
        'Boston PD': 'Boston Police',
        'Boston Library': 'the Boston Public Library',
        'Black Lives Matter': 'BLM',
        'Bay Windows News': 'Bay Windows',
        'South End News': 'the South End News',
        'NECN': 'New England Cable News',
        'GBH Children’s Programming': 'GBH Kids',
        'Under the Radar Program': 'Under the Radar',
        'Basic Black Program': 'Basic Black',
        'WH': 'White House',
        'GOP': 'the Republican party',
        'Patriots': 'the New England Patriots',
        'PS': 'Public Schools',
        'Mission Hill': 'Mission Hill School',
        'Hinckley Allen': 'Hinckley Allen Snyder LLP',
        'Mission Hill': 'Mission Hill street',
        'MA DESE': 'The Massachusetts Department of Elementary and Secondary Education',
        'U.S. Capitol': 'Capitol',
        'BPR': 'Boston Public Radio',
        'Congress': 'U.S. Congress',
        'Mass Avenue': 'Mass. Ave',
        'Melnea Cass': 'Melnea Cass Boulevard',
        'Newport Street': 'Newport',
        'Massachusetts Avenue': 'Mass Avenue',
        'Mass. Ave & Cass': 'Mass and Cass',
        'GBH World': 'GBH WORLD Channel',
        'WGBH Foundation': 'WGBH Educational Foundation'
    }
    for abbr, full in abbreviation_map.items():
        location = re.sub(r'\b' + abbr + r'\b', full, location)
    
    # Handle common synonyms or variants
    synonym_map = {
        'ny': 'new york',
        'la': 'los angeles',
        'sf': 'san francisco',
        'WGBH': 'GBH',
        'Harvard-Smithsonian Astrophysics': 'the Harvard-Smithsonian Center for Astrophysics',
        'Granite Broadcasting Holdings': 'Granite Broadcasting',
        'the Mystic Aquarium': 'Mystic Aquarium',
        'North Shore N.E. Aquarium': 'North Shore Aquarium',
        'Blurb.com': 'Blurb',
        'Sweetwater Co.': 'Sweetwater',
        'Beverly Police': 'Beverly PD',
        'Public Broadcasting Service': 'PBS',
        'ITV plc': 'ITV',
        'ITV Global Entertainment Ltd': 'ITV Global',
        'GBH 2': 'GBH Channel 2',
        'BBC Two': 'BBC 2',
        'the Parabola Center': 'Parabola',
        'Treez of Lyfe': 'Treez',
        'NOVA': 'NOVA PBS',
        'Shutterstock': 'Shutterstock Images',
        'bezikus Eky Studio': 'Eky Studio',
        'Associated': 'Associated Press',
        'Newton Marriott': 'Newton Marriott Hotel',
        'AIM': 'AIM (Alternative Investment Market)',
        'MassReconnect': 'MassReconnect Program',
        'McKinsey Company': 'McKinsey & Company',
        'NPR': 'National Public Radio',
        'Purdue Pharma': 'Purdue',
        'Johnson Johnson McKesson': 'Johnson & Johnson, McKesson',
        'Walmart': 'Wal-Mart',
        'the Justice Department': 'DOJ',
        'the Centers for Disease Control and Prevention': 'CDC',
        'GBH Studio': 'GBH Studios',
        'the Boston Public Library': 'Boston Library',
        'the New England Conservatory Fellowship String Quartet': 'NEC Quartet',
        'Social Security': 'SSA',
        'the Energy Department': 'DOE',
        'the Center on Budget and Policy Priorities': 'CBPP',
        'MBTA': 'T',
        'the Orange Line': 'Orange Line MBTA',
        'GBH News': 'GBH Newsroom',
        'Northeastern University': 'Northeastern',
        'the Red Line': 'Red Line MBTA',
        'GBH Fraser Performance Studio': 'Fraser Studio',
        'GBH Music': 'GBH Music Studio',
        'Unique Music Adventure Rasa Quartet': 'Rasa Quartet',
        'Boston Baroque': 'Baroque',
        'The Rasa String Quartet': 'Rasa Quartet',
        'Concerto Grosso': 'Grosso',
        'Harmonia Artificioso': 'Harmonia',
        'Boston Globe': 'Globe',
        'U.S. Senate': 'Senate',
        'Associated Press': 'AP',
        'the Nevada Independent': 'Nevada Independent',
        'Department of Elementary and Secondary Education': 'DESE',
        'Boston Public Schools': 'BPS',
        'the Boston Public Schools': 'Boston Schools',
        'the Department of Homeland Security': 'DHS',
        'McDonald': 'McDonald\'s',
        'City Council': 'Council',
        'K12 Security Information Exchange': 'K12 Security',
        'the U.S. Senate Committee on Homeland Security and Governmental Affairs': 'Senate Homeland Security Committee',
        'Worcester City Council': 'Worcester Council',
        'the Boston Public Library': 'Boston Library',
        'Nubian Square': 'Nubian',
        'Copley Square': 'Copley',
        'the Sunrise Movement Socialist Alternative': 'Sunrise Movement',
        'the Democratic Socialist party': 'Democratic Socialists',
        'the Department of Public Health': 'DPH',
        'COVID': 'COVID-19',
        'Boston City Hall': 'Boston Hall',
        'Boston Public Radio': 'BPR',
        'Boston Police': 'Boston PD',
        'the Boston Globe': 'Globe',
        'City Council': 'Council',
        'GBH News': 'GBH Newsroom',
        'Boston Public Radio': 'BPR',
        'Centers for Disease Control': 'CDC',
        'the Department of Family Medicine': 'Family Medicine Department',
        'Boston Medical Center': 'BMC',
    }

    for synonym, full in synonym_map.items():
        location = re.sub(r'\b' + synonym + r'\b', full, location)
    
    # Remove extra spaces
    location = re.sub(r'\s+', ' ', location)
    
    return location

# Check if two locations are the same
def are_same_location(loc1, loc2, threshold=85):
    similarity = fuzz.token_set_ratio(loc1, loc2)
    return similarity >= threshold

# Check if a location can be added to the list of locations
def can_add_location(location, locations):
    unwanted_entities = load_cache("./data_prod/unwanted_locations.json")

    if location in unwanted_entities:
        return False
    
    for loc2 in locations:
        if are_same_location(location, loc2):
            return False
    
    return True

# Try finding locations from title
def get_valid_title_locations(header):
    known_title_locs = load_cache("./data_prod/known_locs.json")
    known_locations = known_title_locs.keys()

    locations_list = []
    for location in known_locations:
        loc = normalize_location(location)
        if (loc in header and can_add_location(loc, locations_list)):
            locations_list.append(loc)
    
    if (len(locations_list) == 0):
        return None
    else:
        return locations_list
    
# Return all valid facilities and organizations found
def get_valid_entities(entities):    
    valid_facilities = []
    valid_orgs = []

    # TODO: Limit the number of entities to consider
    # TODO: Check frequency and order them by popularity
    for entity in entities:
        loc = normalize_location(entity.text)
        if (entity.label_ == "FAC"):
            if (can_add_location(loc, valid_facilities)):
                valid_facilities.append(loc)
        elif (entity.label_ == "ORG"):
            if (can_add_location(loc, valid_orgs)):
                valid_orgs.append(loc)
    
    valid_entities = valid_facilities + valid_orgs

    if (len(valid_entities) == 0):
        return None
    else: 
        return valid_entities
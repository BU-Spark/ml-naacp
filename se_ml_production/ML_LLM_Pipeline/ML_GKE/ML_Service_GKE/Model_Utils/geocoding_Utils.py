import requests
from Model_Utils.helper_functions import load_cache, save_cache, check_time

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
            state = results['result']['geographies']['2020 Census Blocks'][0]['STATE']
            
            return tract, county, state
        except IndexError:
            print("[ERROR] Unable to retrieve census geography for: " + location)
        except KeyError:
            print("[ERROR] Location is outside of the United States: " + location)
        except Exception as error:
            print(f"[ERROR] Error retrieving census geography for: {location}, Error: {error}")

    print("[ERROR] API call failed for: " + location + " with coordinates" + str(coordinates))
    return None, None, None  # Return this if API call failed or no tracts found

# Get the census tract and county of the location
def geocode(location, coordinates):
    known_locations_path = "./data_prod/known_locations.json"
    known_locations = load_cache(known_locations_path)

    if (location is None or len(location) == 0): return None, None, None, None  

    if (coordinates is None or len(coordinates) == 0
        or coordinates[0] is None or coordinates[1] is None): return None, None, None, None  

    # Only geocode if it's not known
    Tract = known_locations[location]["tract"]
    County = known_locations[location]["county"]
    State = known_locations[location]["state"]
    City = known_locations[location]["city"]

    if (Tract is None or County is None or State is None):
        # Geocode article
        coordinates = known_locations[location]["coordinates"]
        Tract, County, State = query_census_api(location, coordinates)

        # Save to cache
        known_locations[location]["tract"] = Tract
        known_locations[location]["county"] = County
        known_locations[location]["state"] = State
        save_cache(known_locations, known_locations_path)
    
    return Tract, County, State, City
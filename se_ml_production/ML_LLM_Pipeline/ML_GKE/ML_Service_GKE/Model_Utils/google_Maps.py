import googlemaps
import json
from Model_Utils.helper_functions import save_cache

class GoogleMapsClient:
    """
    A wrapper to manage the Google Maps API.
    """
    def __init__(self):
        self.client = None

    # Create a Google Maps Client
    def createMapsClient(self, API_KEY=None):
        try:
            if (API_KEY == None):
                raise Exception("No API Key Given!")
            self.client = googlemaps.Client(key=API_KEY)
        except Exception as e:
            print(f"Failed to create Google Maps Client! {e}")
            raise Exception("Fatal Error in creating Google Maps Client.")
        return
    
    # Function to get the center coordinates of a city/state. These are used to discard 'bad' locations
    def getCenterCoords(self, location):
        gmaps = self.client
        try:
            # Locations are limited to Massachusetts for now
            geocode_result = gmaps.geocode(f"{location}", components={"country": "US"})
            
            if (len(geocode_result) > 0):
                longitude = geocode_result[0]['geometry']['location']['lng']
                latitude = geocode_result[0]['geometry']['location']['lat']
                
                return [longitude, latitude]
            else:
                print(f"[WARNING] Could not find location {location} through google maps")
                return [None, None]
        except Exception as error:
            print(f"[ERROR] Error finding locations through google maps, {error}")
            return [None, None]
    
    # Check if a location is'bad', meaning it's the center of a city/state
    def isCenter(self, coords, entity):
        centers_path = "./data_prod/centers.json"
        try: 
            with open(centers_path, 'r') as file:
                centers_cache = json.load(file)
        except FileNotFoundError:
            centers_cache = {}
        
        
        if entity:
            if entity in centers_cache:
                if centers_cache[entity] == coords: return True
                
                entity_coords = self.getCenterCoords(entity)
                centers_cache[entity] = entity_coords
                save_cache(centers_cache, centers_path)
                if entity_coords == coords: return True
        return False 
    
    # Unpack the geocode result to get the coordinates, city, state, and location type
    def get_info(self, geocode_result):
        longitude = geocode_result[0]['geometry']['location']['lng']
        latitude = geocode_result[0]['geometry']['location']['lat']
        coords = [longitude, latitude]

        # Get the city from the address components
        address_components = geocode_result[0]['address_components']
        city, state = None, None
        for component in address_components:
            if 'locality' in component['types']:
                city = component['long_name']
            elif 'administrative_area_level_1' in component['types']:
                state = component['long_name']
        
        # Check if the location is invalid
        location_type = geocode_result[0]["geometry"]["location_type"]
        return coords, city, state, location_type
    
    # Google Maps API handler
    def callGoogleMapsAPI(self, location):
        gmaps = self.client
        try:
            # Locations are limited to Massachusetts for now
            geocode_result = gmaps.geocode(f"{location}", components={"country": "US"})
            
            if (len(geocode_result) > 0):
                coords, city, state, location_type = self.get_info(geocode_result)
                longitude, latitude = coords
                
                if location_type == "APPROXIMATE" or self.isCenter(coords, city) or self.isCenter(coords, state):
                    return None, None, None
                
                return longitude, latitude, city
            else:
                print(f"[WARNING] Could not find location {location} through google maps")
                return None, None, None
        except Exception as error:
            print(f"[ERROR] Error finding locations through google maps, {error}")
            return None, None, None

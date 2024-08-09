import googlemaps

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
    
    # Make a call to the Google Maps API to get the coordinates of the location
    def getCoordinates(self, location, state="Massachussetts", components={"administrative_area_level": "MA", "country": "US"}):
        gmaps = self.client
        try:
            geocode_result = gmaps.geocode(f"{location}, {state}", components=components)
            if (len(geocode_result) > 0):
                longitude = geocode_result[0]['geometry']['location']['lng']
                latitude = geocode_result[0]['geometry']['location']['lat']
                return longitude, latitude
            else:
                return None
        except Exception as error:
            print(f"Failed to get coordinates of location {location}, with error: ", error)
            return None


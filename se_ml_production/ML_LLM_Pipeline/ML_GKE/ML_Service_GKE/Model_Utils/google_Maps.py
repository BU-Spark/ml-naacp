import googlemaps

class GoogleMapsClient:
    """
    A wrapper to manage the Google Maps API.
    """
    def __init__(self):
        self.client = None

    def createMapsClient(self, API_KEY=None):
        try:
            if (API_KEY == None):
                raise Exception("No API Key Given!")
            self.client = googlemaps.Client(key=API_KEY)
        except Exception as e:
            print(f"Failed to create Google Maps Client! {e}")
            raise Exception("Fatal Error in creating Google Maps Client.")
        return
import requests

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
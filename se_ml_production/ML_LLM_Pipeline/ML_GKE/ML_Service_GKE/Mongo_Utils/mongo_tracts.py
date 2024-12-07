import secret
import requests

from Mongo_Utils.mongo_funcs import get_collection

from global_state import global_instance

# Get census demographics for any given article
def get_census_demographics(year, dsource, dname, tract, county, state):
    cols = 'NAME,P2_001N,P2_002N,P2_003N,P2_004N,P2_005N,P2_006N,P2_007N,P2_008N,P2_009N,P2_010N'
    base_url = f"https://api.census.gov/data/{year}/{dsource}/{dname}"

    census_url = f"{base_url}?get={cols}&for=tract:{tract}&in=county:{county}&in=state:{state}"

    census_response = requests.get(census_url)
    census_response_json = census_response.json()

    return census_response_json

def get_city_demographics(year, dsource, dname, city, state):
    cols = 'NAME,P2_001N,P2_002N,P2_003N,P2_004N,P2_005N,P2_006N,P2_007N,P2_008N,P2_009N,P2_010N'
    base_url = f"https://api.census.gov/data/{year}/{dsource}/{dname}"

    # Note: Adjust 'for' and 'in' parameters based on city-level geography
    census_url = f"{base_url}?get={cols}&for=place:*&in=state:{state}"

    census_response = requests.get(census_url)
    census_response_json = census_response.json()
    
    # Filter results to find the specific city
    city_demographics = [
        item for item in census_response_json[1:]
        if city in item[0]  # Assuming the city name is in the first column of the results
    ]
    columns = cols.split(",")
    city_demographics = [columns, city_demographics[0]]
    return city_demographics

def update_demographics(tract_collection, tract, county, state, city=None):
    try:
        if city:
            census_data = get_city_demographics("2020", "dec", "pl", city, state)
        else:
            census_data = get_census_demographics("2020", "dec", "pl", tract, county, state)
        
        if not census_data or len(census_data) < 2:
            raise ValueError("Census data is missing or malformed.")
        
        headers = census_data[0]  # Headers
        values = census_data[1]   # Data values
        data = dict(zip(headers, values))

        county_name = data.get('NAME', "")
        geoid_tract = f"{state}{county}{tract}"

        # Prepare the update document
        update_doc = {
            'demographics.p2_001n': str(data.get('P2_001N', 0)),
            'demographics.p2_002n': str(data.get('P2_002N', 0)),
            'demographics.p2_003n': str(data.get('P2_003N', 0)),
            'demographics.p2_004n': str(data.get('P2_004N', 0)),
            'demographics.p2_005n': str(data.get('P2_005N', 0)),
            'demographics.p2_006n': str(data.get('P2_006N', 0)),
            'demographics.p2_007n': str(data.get('P2_007N', 0)),
            'demographics.p2_008n': str(data.get('P2_008N', 0)),
            'demographics.p2_009n': str(data.get('P2_009N', 0)),
            'demographics.p2_010n': str(data.get('P2_010N', 0)),
            'county_name': county_name,
            'geoid_tract': geoid_tract
        }
        
        # Update MongoDB document
        tract_collection.update_one(
            {'tract': tract},
            {'$set': update_doc}
        )

    except Exception as error:
        print(f"[ERROR] Error getting census data for tract {tract}: {error}")
        tract_collection.update_one(
            {'tract': tract},
            {'$set': {"canFind": False}}
        )
        return

# Update an existing tract document that did not have demographics data
def update_tracts(tract_collection, tract, neighborhood, county, state, city, article, location):    
    empty_doc = {
        'tract': tract,
        'state': state,
        'county': county,
        'city': city,
        'neighborhood': neighborhood,
        'county_name': "",
        'geoid_tract': "",
        'demographics': {},
        'articles': [article],
        'locations': [location],
        'canFind': True
    }
    tract_collection.insert_one(empty_doc)
    update_demographics(tract_collection, tract, county, state)

# Create a new tract document in the database and add its demographics data
def create_gen_tract(tract_collection, tract, neighborhood, county, state, city, article, location):    
    empty_doc = {
        'tract': tract,
        'state': state,
        'county': "",
        'city': city,
        'neighborhood': neighborhood,
        'county_name': "",
        'geoid_tract': "",
        'demographics': {},
        'articles': [article],
        'locations': [location],
        'canFind': True
    }
    tract_collection.insert_one(empty_doc)
    update_demographics(tract_collection, tract, county, state, city)
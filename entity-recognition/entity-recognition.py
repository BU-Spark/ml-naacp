import pandas as pd
import en_core_web_sm
import spacy
import googlemaps 
import requests

import secret


def get_locations(article_text):
    """
    get location names from article using NER 
    input: article_text as a string, aggregate of h1, h2, lede, and body
    returns: locations - set of tuples of (NAME, 'GPE')
    """
    # get locations using NER  
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(article_text)
    print(doc.ents)
    # get the locations only, remove duplicates from results 
    locations = set([(X.text, X.label_) for X in doc.ents if X.label_ == 'GPE' or X.label_ == 'FAC' or X.label_ == 'ORG']) # or X.label_ == 'LOC' or X.label_ == 'FAC' or X.label_ == 'ORG'
    
    return locations

def get_location_geocode(API_KEY, locations):
    """
    getting coordinates from location names in articles 
    input: google maps platform API KEY, locations article 
    return: dictionary of location names (key) with coordinates (value as a dictionary with lat and lon as keys)
    """
    gmaps = googlemaps.Client(key=API_KEY)
    results = {}

    # getting coordinates
    for place in locations:
        # we can constrain google geocode api search to massachusetts or us - census geocoder will not work for places outside of U.S 
        #geocode_result = gmaps.geocode(place[0] + ", MA, USA") # place is a tuple, where first value is the location name 
        geocode_result = gmaps.geocode(place[0], components={"administrative_area": "MA", "country": "US"})
        temp = {}
        try:
            temp['lat'] = geocode_result[0]['geometry']['location']['lat']
            temp['lon'] = geocode_result[0]['geometry']['location']['lng']
            results[place[0]] = temp
        except IndexError: # unable to get coordinates for location
            print("Unable to locate " + place[0])

    return results 


# links: https://www2.census.gov/geo/pdfs/maps-data/data/GeocodingURL.pdf
#        https://geocoding.geo.census.gov/geocoder/Geocoding_Services_API.pdf
def get_census_geos(geocode_results):
    """
    get census geographies - tract, block group, block by coordinates
    input: google maps geocode_results as a dictionary
    return: block, block_group, tract, county for each location
    """
    census_geos = {}
    for place in geocode_results:
        # building the geocoding url
        base_url = f'https://geocoding.geo.census.gov/geocoder/geographies/coordinates?'
        survey_ver = f'&benchmark=4&vintage=4&layers=2020 Census Blocks&format=json'
        lon = geocode_results[place]['lon']
        lat = geocode_results[place]['lat']
        census_geo_url = f'{base_url}x={lon}&y={lat}{survey_ver}'

        # getting the census geographies 
        response = requests.get(census_geo_url)
        response_json = response.json()

        try:
            block = response_json['result']['geographies']['2020 Census Blocks'][0]['BLOCK']
            block_group = response_json['result']['geographies']['2020 Census Blocks'][0]['BLKGRP']
            tract = response_json['result']['geographies']['2020 Census Blocks'][0]['TRACT']
            county = response_json['result']['geographies']['2020 Census Blocks'][0]['COUNTY']
            census_geos[place] = {'block': block,
                                  'blkgrp': block_group,
                                  'tract': tract,
                                  'county': county}
        except IndexError:
            print("Unable to retrieve census geography for: " + place)
        except KeyError:
            print("Location is outside of the United States: " + place)
    return census_geos



"""
# CENSUS DATA API 
# https://www.census.gov/content/dam/Census/library/publications/2020/acs/acs_api_handbook_2020_ch02.pdf
# any user can query small quantities of data with minimal restrictions - up to 50 variables in a single query, up to 500 queries per IP address per day 
# more than 500 queries per IP address per day requires you to register for API key - www.census.gov/developers
# https://www.census.gov/data/developers/data-sets/decennial-census.html 
"""
def get_census_demographics(year, dsource, dname, tract, county, state):
    # input: census year, data source, survey name, tract, county, state
    # return: demographic data for tract mentioned
    
    # census variables: https://api.census.gov/data/2020/dec/pl/variables.html 
    cols = 'NAME,P2_001N,P2_002N,P2_003N,P2_004N,P2_005N,P2_006N,P2_007N,P2_008N,P2_009N,P2_010N'
    base_url = f"https://api.census.gov/data/{year}/{dsource}/{dname}"

    # to get tract demographics 
    census_url = f"{base_url}?get={cols}&for=tract:{tract}&in=county:{county}&in=state:{state}"

    # to get block demographics 
    # census_url = f"{base_url}?get={cols}&for=block:{block}&in=tract:{tract}&in=county:{county}&in=state:{state}"

    census_response = requests.get(census_url)
    census_response_json = census_response.json()

    return census_response_json

def run_pipeline(text, year, dsource, dname, state, API_KEY):
    locations = get_locations(text)
    #locations = {('Boston', 'GPE'), ('Massachusetts', 'GPE'), ('Boston city', 'GPE'), ('Roxbury', 'GPE'), ('Fitchburg', 'GPE'), ('Medford', 'GPE')}
    print(locations)
    #location_geocode = get_location_geocode(API_KEY, locations)
    #location_geocode = {'Boston': {'lat': 42.3600825, 'lon': -71.0588801}, 'Massachusetts': {'lat': 42.4072107, 'lon': -71.3824374}, 'Boston city': {'lat': 42.3600825, 'lon': -71.0588801}, 'Roxbury': {'lat': 42.3125672, 'lon': -71.0898796}, 'Fitchburg': {'lat': 42.5834228, 'lon': -71.8022955}, 'Medford': {'lat': 42.4184296, 'lon': -71.1061639}}
    print(location_geocode)
    census_geos = get_census_geos(location_geocode)

    result = []
    for place_name in census_geos:
        
        place_info = {}
        county = census_geos[place_name]['county']
        tract = census_geos[place_name]['tract']

        try:
            demographic_results = get_census_demographics(year, dsource, dname, tract, county, state)

            place_info[place_name] = {'county_code': county} 
            place_info[place_name] = {'county_name': demographic_results[1][0]}
            place_info[place_name]['tract'] = tract 
            place_info[place_name]['demographics'] = {
                'p2_001n': demographic_results[1][1], # total population 
                'p2_002n': demographic_results[1][2], # total hispanic or latino 
                'p2_003n': demographic_results[1][3], # total not hispanic or latino 
                'p2_004n': demographic_results[1][4], # total not hispanic or latino - pop of one race
                'p2_005n': demographic_results[1][5], # total not hispanic or latino - pop of one race - white alone 
                'p2_006n': demographic_results[1][6], # total not hispanic or latino - pop of one race - black or african american alone
                'p2_007n': demographic_results[1][7], # total not hispanic or latino - pop of one race - american indian and alaska native alone
                'p2_008n': demographic_results[1][8], # total not hispanic or latino - pop of one race - asian alone 
                'p2_009n': demographic_results[1][9], # total not hispanic or latino - pop of one race - native hawaiian and other pacific islander alone
                'p2_010n': demographic_results[1][10] # total not hispanic or latino - pop of one race - some other race alone 
            } 
            result.append(place_info)
        except Exception:
            print("Unable to get census demographics for: " + place_name)
    
    return result


def main():
    # we probably won't be reading from a csv but this is just for testing the pipeline with some articles 
    # read data 
    df = pd.read_csv("/Users/mvoong/Desktop/naacp-subneighborhood update/test-data/multiple_article_test.csv")

    # prepare the data 
    i = 5
    h1 = str(df['hl1'][i])
    h2 = str(df['hl2'][i])
    lede = str(df['lede'][i])
    body = str(df['body'][i])
    text = h1 + h2 + lede + body
    article_id = df['content-id'][i]

    # set variables 
    year='2020'
    dsource='dec' # which survey are we interested in ? decennial 
    dname='pl' # a dataset within a survey, pl - redistricting data 
    state='25' # state code 

    result = {}
    temp = run_pipeline(text, year, dsource, dname, state, secret.API_KEY)
    result[article_id] = temp 
    
    return result

if __name__ == "__main__":
    import json
    print(json.dumps(main(),sort_keys=True, indent=2))

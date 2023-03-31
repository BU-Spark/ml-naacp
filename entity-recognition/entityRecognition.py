import pandas as pd
import spacy
import googlemaps 
import requests
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from bs4 import BeautifulSoup 
import nltk

import secret
import censusNeighborhood


def get_locations(article_text):
    """
    get location names from article using NER - spacy 
    input: article_text as a string, aggregate of h1, h2, lede, and body
    returns: locations - set of tuples of (NAME, 'GPE')
    """
    # get locations using NER  
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(article_text)

    # get the locations only, remove duplicates from results 
    locations = set([(X.text, X.label_) for X in doc.ents if X.label_ == 'GPE']) # or X.label_ == 'LOC' or X.label_ == 'FAC' or X.label_ == 'ORG'
    
    return locations

def get_locations_bert(article_text):
    """
    get location names from article using NER - bert model 
    https://huggingface.co/dslim/bert-base-NER
    input: article_text as a string, aggregate of h1, h2, lede, and body
    returns: locations - set of tuples of (NAME, 'LOC') and organizations - set of tuples (NAME, 'ORG) mentioned in the article
    """
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    
    ner_results = nlp(article_text)
    locations = set([(X['word'],X['entity_group']) for X in ner_results if X['entity_group'] == 'LOC'])
    orgs = set([(X['word'], X['entity_group']) for X in ner_results if X['entity_group'] == 'ORG'])

    return locations, orgs


def get_lede(text, num_sent):
    """
    get the lede from the article_text 
    input: headline of the article if it is provided, atricle_text as a string, and num_sent - number of sentences to return 
    returns: headline and first x (num_sent) sentences 
    """
    
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text()
    lede = nltk.sent_tokenize(clean_text)[:num_sent] # returns a list
    lede = ".".join(lede)

    return lede

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
    #locations = get_locations(text)
    locations, orgs = get_locations_bert(text)
    #locations = {('Boston', 'GPE'), ('Massachusetts', 'GPE'), ('Boston city', 'GPE'), ('Roxbury', 'GPE'), ('Fitchburg', 'GPE'), ('Medford', 'GPE')}
    
    location_geocode = get_location_geocode(API_KEY, locations)
    org_geocode = get_location_geocode(API_KEY, orgs)
    #location_geocode = {'Boston': {'lat': 42.3600825, 'lon': -71.0588801}, 'Massachusetts': {'lat': 42.4072107, 'lon': -71.3824374}, 'Boston city': {'lat': 42.3600825, 'lon': -71.0588801}, 'Roxbury': {'lat': 42.3125672, 'lon': -71.0898796}, 'Fitchburg': {'lat': 42.5834228, 'lon': -71.8022955}, 'Medford': {'lat': 42.4184296, 'lon': -71.1061639}}
    print(location_geocode)
    #print(org_geocode)
    #location_geocode = {'Salem': {'lat': 42.5197473, 'lon': -70.8954626}, 'Massachusetts': {'lat': 42.4072107, 'lon': -71.3824374}, 'Salem City Hall': {'lat': 42.5218851, 'lon': -70.8956157}}
    #org_geocode = ""
    census_geos = get_census_geos(location_geocode) #| org_geocode) # combining dictionaries

    result = []
    mappings = censusNeighborhood.neighborhood_mapping()
    for place_name in census_geos:
        place_info = {}
        county = census_geos[place_name]['county']
        tract = census_geos[place_name]['tract']
        print(tract)

        if mappings.tract_mapping[state + county + tract]: # get corresponding neighborhood
            place_info[place_name]['neighborhood'] = mappings.tract_mapping[state + county + tract]
            try:
                demographic_results = get_census_demographics(year, dsource, dname, tract, county, state)

                # build result dictionary 
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
            except Exception as e:
                print(e)
                print("Unable to get census demographics for: " + place_name)
    return result


def main():
    # we probably won't be reading from a csv but this is just for testing the pipeline with some articles 
    # read data 
    df = pd.read_csv("./gbh_rss/gbh-rss-test.csv")

    # get lede (first x number of sentences)

    # set variables 
    year='2020'
    dsource='dec' # which survey are we interested in ? decennial 
    dname='pl' # a dataset within a survey, pl - redistricting data 
    state='25' # state code 

    result = {}
    idx = 0
    text = str(df['description'][idx]) + str(df['content'][idx])
    text = get_lede(text, 5)
    temp = run_pipeline(text, year, dsource, dname, state, secret.API_KEY)
    result[df['UID'][idx]] = temp
    """
    for idx in range(len(df)):
        text = str(df['description'][idx]) + str(df['content'][idx])
        text = get_lede(text, 5)
        temp = run_pipeline(text, year, dsource, dname, state, secret.API_KEY)
        result[df['UID'][idx]] = temp
    """
    return result


if __name__ == "__main__":
    import json
    test = main()
    print(json.dumps(test,sort_keys=True, indent=2))

    with open('gbh-sample-test.json', 'w') as fp:
        json.dump(test, fp)
    

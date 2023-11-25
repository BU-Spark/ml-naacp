import re
import json 
import nltk
import spacy
import requests
import googlemaps 
import pandas as pd
from bs4 import BeautifulSoup

def get_locations_bert(article_text, nlp):
    """
    get location names from article using NER - bert model 
    https://huggingface.co/dslim/bert-base-NER
    input: article_text as a string, aggregate of h1, h2, lede, and body
    returns: locations - set of tuples of (NAME, 'LOC') and organizations - set of tuples (NAME, 'ORG) mentioned in the article
    """
    ner_results = nlp(article_text)
    locations = set([(X['word'],X['entity_group']) for X in ner_results if X['entity_group'] == 'LOC'])
    orgs = set([(X['word'], X['entity_group']) for X in ner_results if X['entity_group'] == 'ORG'])
    return locations, orgs

def clean_entity_results(extracted_loc, extracted_orgs, drop_geos):
    # cleaning extracted entities from bert 
    # removing state names, and mass town names since the demographics data is too broad
    # return cleaned set of entities
    entity_result = extracted_loc | extracted_orgs

    for tup in extracted_loc | extracted_orgs:
        if len(tup[0]) <= 1:
            entity_result.remove(tup)
        elif tup in drop_geos.state_entities:
            entity_result.remove(tup)
        elif tup in drop_geos.mass_town_entities:
            entity_result.remove(tup)
        elif tup in drop_geos.org_entities:
            entity_result.remove(tup)
    return entity_result

def remove_existing_geocodes(entity_result, saved_geocodes):
    # check if any locations or organizations were recognized
    # check if the geocodes already exist in dictionary
    existing_loc_geocode = {}
    new_loc_geocode = set()
    for ent in entity_result:
        try:
            existing_loc_geocode[ent[0]] = saved_geocodes[ent[0]]
        except KeyError:
            new_loc_geocode.add(ent)
    return existing_loc_geocode, new_loc_geocode

def get_snippet(sentences, num_sent, lede=True, remaining_text=False):
    """
    get the snippet of text from the article_text, replace single quotes
    input: article text, and num_sent - number of sentences to return, default lede is true will return first x sentences
           reamaining_text then must be False 
    returns: first x (num_sent) sentences
    """
    #clean_text = clean_article_text(text)
    #clean_text = ". ".join(clean_text.split(".")) # adding a space after period so nltk can do a better job recognizing sentences
    #lede = nltk.sent_tokenize(clean_text)[:num_sent] # returns a list
    
    if lede: # get the first num_sent 
        lede_text = sentences[:num_sent]
        result_text = " ".join(lede_text)
    elif remaining_text: # get rest of article num_sent * 2 until the end
        result_text = sentences[num_sent*2:]
        result_text = " ".join(result_text)
    else: # get sentences num_sent to num_sent * 2
       result_text = sentences[num_sent:num_sent*2]
       result_text = " ".join(result_text) 
    
    singleq = result_text.replace('’', "'")

    return singleq

def get_sentences(text):
    # return article text as a list of its sentences 

    clean_text = clean_article_text(text)
    clean_text = ". ".join(clean_text.split(".")) # adding a space after period so nltk can do a better job recognizing sentences
    sentences = nltk.sent_tokenize(clean_text)

    return sentences

def clean_article_text(text):
    # get text, removing html tags
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text()
    return clean_text

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
        #geocode_result = gmaps.geocode(place[0] + ", Suffok County, MA, USA") # place is a tuple, where first value is the location name 
        geocode_result = gmaps.geocode(place[0] + ", Suffolk County",  components={"administrative_area_level": "MA", 
                                                                                   "country": "US"})
        #print(geocode_result)
        #print()
        temp = {}
        try:
            geocode_components = geocode_result[0]['address_components']
            for i, addr_comp in enumerate(geocode_components):
                if 'administrative_area_level_2' in addr_comp['types']:
                    if "Suffolk County" == addr_comp['short_name'] and i != 0:
                        temp['lat'] = geocode_result[0]['geometry']['location']['lat']
                        temp['lon'] = geocode_result[0]['geometry']['location']['lng']
                        results[place[0]] = temp
                        
        except IndexError: # unable to get coordinates for location
            print("Unable to locate " + place[0])

    return results 

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

def run_entity_recognition(text, nlp, drop_geos, saved_geocodes):
    # running entity recogntion on text
    # parse existing geocoded entities and new geocoded entities
    try:
        extracted_loc, extracted_orgs = get_locations_bert(text, nlp)
        ent_result = clean_entity_results(extracted_loc, extracted_orgs, drop_geos)
        existing_loc_geocode, new_loc_geocode = remove_existing_geocodes(ent_result, saved_geocodes)
    except TypeError as e:
        print(f"No entities: {e}")
        existing_loc_geocode = {}
        new_loc_geocode = set()

    return existing_loc_geocode, new_loc_geocode

def run_location_geocode(API_KEY, new_loc_geocode):
    # get geocodes for NEW locations and saving them to json
    # returns new location geocodes as dictionary 
    location_geocode = {}
    if new_loc_geocode:
        location_geocode = get_location_geocode(API_KEY, new_loc_geocode)
    return location_geocode

def check_snippets(API_KEY, new_entities, existing_entities):
    location_geocode = run_location_geocode(API_KEY, new_entities)
    existing_loc_geocode = existing_entities
    combined_geocodes = location_geocode | existing_loc_geocode # if this is empty, then try the next snippet of text 
    return (not combined_geocodes), location_geocode, existing_loc_geocode

def run_pipeline(year, dsource, dname, state, existing_loc_geocode, location_geocode, mappings):
    census_geos = get_census_geos(location_geocode | existing_loc_geocode)

    result = []
    for place_name in census_geos:
        place_info = {}
        county = census_geos[place_name]['county']
        tract = census_geos[place_name]['tract']
        
        try:
            demographic_results = get_census_demographics(year, dsource, dname, tract, county, state)

            # build result dictionary 
            place_info[place_name] = {'county_code': county} 
            place_info[place_name] = {'county_name': demographic_results[1][0]}
            place_info[place_name]['tract'] = tract
            geoid_tract = state + county + tract # this includes the state and county and tract number
            place_info[place_name]['geoid_tract'] = geoid_tract

            if mappings.tract_mapping.get(geoid_tract): # get corresponding boston neighborhood 
                place_info[place_name]['neighborhood'] = mappings.tract_mapping[state + county + tract]
            
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











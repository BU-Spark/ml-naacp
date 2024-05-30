import os
import json
import zipfile
from art import *
from bson import ObjectId
from pymongo import MongoClient
from google.cloud import storage

from Model_Utils.model_Loaders import load_bert_NER, load_bert_TOPIC

from Utils.spinner import Spinner
from NER_Classes.censusClass import census_data
from NER_Classes.geographyEntities import geography
from Mongo_Utils.mongoDB_manager import MongoDBManager
from NER_Classes.censusNeighborhood import neighborhood_mapping

def fetch_llm_models():
	"""
	Runs through google cloud bucket to fetch the appropriate LLM's for prediction
	"""
	try:
		storage_client = storage.Client()
		bucket = storage_client.bucket("naacp-models")
		blob = bucket.blob("BERTopic_Models/bglobe_519_body_230_cereal.zip")
		blob.download_to_filename("./llm_models/bglobe_519_body_230_cereal.zip")

		destination_file_name = "./llm_models/bglobe_519_body_230_cereal.zip"
		with zipfile.ZipFile(destination_file_name, 'r') as zip_ref: # Extract the ZIP
			zip_ref.extractall("./llm_models")
		os.remove(destination_file_name) # Remove the ZIP for cleanup
	except Exception as e:
		print(f"Model fetching failed! {e}")
		raise Exception("Fatal Error in Fetching LLM models. Exiting...")
	return

def load_heirarch_data(db, fetch=True):
	"""
	Loads the Heirarchal data for mapping given after the topic model
	"""
	try:
		if (db == None):
			raise Exception("No database was given!")
		if (fetch):
			bucket = db.bucket("ml_naacp_model_data")
			blob = bucket.blob("Topic_Modeling_Pipeline_Data/openai_label_from_taxonomy_structured_230.json")
			blob.download_to_filename("./geodata_prod/openai_label_from_taxonomy_structured_230.json")

		heirarch_db = json.load(open("./geodata_prod/openai_label_from_taxonomy_structured_230.json"))
        
		if (heirarch_db == None):
			raise Exception(f"heirarch_db returned None!")
            
		return heirarch_db
	except Exception as err:
		print(f"Error loading hierarchal data!\nError: {err}")
		raise Exception("Fatal Error in fetching hierarchal data!")
	return

def bootstrap_pipeline():
	"""
	Bootstraps the entire pipeline for inference
	"""
	try:
		file_manifest = [
			"tracts-neighbors.json", # Neigh Mappings
			"blocks-neighbors.json", # Neigh Mappings
			"saved-geocodes.json", # Geography
			"states.csv", # Geography
			"mass-towns.csv", # Geography
			"openai_label_from_taxonomy_structured_230.json", # Topic Modeling
			"census_2020_neigh.csv", # Census
			"census_2020.csv", # Census
			"census.json" # Census
		]
		dependency_resolver_arr = []
        
		tprint("BU Spark!", font="sub-zero")
		print("Bootstrapping pipeline... (This should only run once!)")
		print()
		spinner = Spinner("Setting up variables...")
		spinner.start()
		year='2020'
		dsource='dec' # which survey are we interested in ? decennial 
		dname='pl' # a dataset within a survey, pl - redistricting data 
		state='25' # state code 
		spinner.stop()
		print()

		spinner = Spinner("Creating MongoDB Manager..")
		spinner.start()
		db_manager = MongoDBManager()
		spinner.stop()
		print()

		spinner = Spinner("Connecting to Google Cloud Storage Bucket...")
		spinner.start()
		db = storage.Client()
		spinner.stop()
		print()

		spinner = Spinner("Detecting & Making local directories for Models...")
		spinner.start()
		spinner.stop()
		llm_model_directory_path = "./llm_models"
		if (not os.path.exists(llm_model_directory_path)):
			print(f"No {llm_model_directory_path}! Creating...")
			os.makedirs(llm_model_directory_path)
			spinner = Spinner("Fetching models...")
			spinner.start()
			fetch_llm_models()
			spinner.stop()
		else:
			print(f"Found! {llm_model_directory_path}!")
			spinner = Spinner("Validating model files...")
			spinner.start()
			if (not os.path.isfile("./llm_models/bglobe_519_body_230_cereal")):
				spinner.stop()
				spinner = Spinner("Model file not found! Pulling models...")
				spinner.start()
				fetch_llm_models()
				spinner.stop()
			else:
				spinner.stop()

		print("Model files successfully validated.\n")

		spinner = Spinner("Detecting & Making local directories for Geographic Data...")
		spinner.start()
		spinner.stop()
		geodata_directory_path = "./geodata_prod"
		if (not os.path.exists(geodata_directory_path)):
			print(f"No {geodata_directory_path}! Creating...")
			os.makedirs(geodata_directory_path)
			dependency_resolver_arr = len(file_manifest) * [True]
		else:
			print(f"Found! {geodata_directory_path}!")
			spinner = Spinner("Validating geodata files...")
			spinner.start()
			for file in file_manifest:
				if (not os.path.isfile(f"./geodata_prod/{file}")):
					dependency_resolver_arr.append(True)
				else:
					dependency_resolver_arr.append(False)
			spinner.stop()
		print("Geodata files successfully validated and updated dependency array.\n")
        
		spinner = Spinner("Fetching OpenAI Hierarchal Mappings...")
		spinner.start()
		heir_data = load_heirarch_data(
			db,
			fetch=dependency_resolver_arr[5]
		)
		spinner.stop()
		print()

		spinner = Spinner("Instantiating classes...")
		spinner.start()
		mappings = neighborhood_mapping(
			db=db,
			fetch_arr=dependency_resolver_arr[:2]
		)
		drop_geos = geography(
			db=db,
			fetch_arr=dependency_resolver_arr[2:5] 
		)
		census_base = census_data(
			db=db,
			fetch_arr=dependency_resolver_arr[6:] 
		)
		saved_geocodes = drop_geos.saved_geocodes 
		spinner.stop()
		print()
        
		spinner = Spinner("Loading Models...")
		spinner.start()
		nlp_ner = load_bert_NER()
		nlp_topic = load_bert_TOPIC()
		spinner.stop()
        
		print("\nBootstrap complete!\n")
		return year, dsource, dname, state, drop_geos, mappings, census_base, heir_data, saved_geocodes, nlp_ner, nlp_topic, db, db_manager
	except Exception as e:
		spinner.err()
		print(f"Bootstrap Failed!!!\nFatal Error:{e}")
		raise Exception("Fatal Error in Bootstrapping ML Pipeline. Exiting...")
        
	return

def validate_bootstrap(year, dsource, dname, state, drop_geos, mappings, census_base, heir_data, saved_geocodes, nlp_ner, nlp_topic, db, db_manager):
	"""
	Validates the bootstrap variables, checking if they exist.
	"""
	try:
		spinner = Spinner("Validating Bootstrap variables...")
		spinner.start()

		variable_manifest = {
			"year":year, 
			"dsource": dsource, 
			"dname": dname, 
			"state": state,
			"drop_geos": drop_geos,
			"mappings": mappings,
			"census_base": census_base,
			"heir_data": heir_data,
			"saved_geocodes": saved_geocodes, 
			"nlp_ner": nlp_ner, 
			"nlp_topic": nlp_topic,
			"db": db,
			"db_manager": db_manager
		}
        
		for var in variable_manifest.keys():
			if (variable_manifest[var] == None):
				raise Exception(f"{var} returned None!. Exiting...")
		spinner.stop()
		print()
		print("Validation Complete! Everything seems to be in order.\n")
	except Exception as e:
		spinner.err()
		print(f"Bootstrap Validation Failed!!!\nFatal Error:{e}")
		raise Exception("Fatal Error in Bootstrapping Validation. Exiting...")

### Mongo DB Bootstrappers

def bootstrap_MongoDB_Prod(client, defined_collection_names):
    """
    Adds the upload collection and other necessities that both the GraphQL and AI Pipeline share.
    Sets up the databse.
    """
    try:
        spinner = Spinner("Checking and Bootstrapping Production DB...\n")
        spinner.start()
        if (client == None):
            raise Exception("No database was given!")

        db_prod = client[os.environ['db_name']]
            
        # Here we check for the upload collection and make it if it doesn't exist
        collection_list = db_prod.list_collection_names()
        for collection in defined_collection_names:
            if collection not in collection_list:
                db_prod.create_collection(collection)
                print(f"[INFO] Collection '{collection}' created.\n")
        spinner.stop()
		print("Deployment Test Comlpete with no erors")
    except Exception as err:
        spinner.err()
        print(f"[Error!] Error in Bootstrapping MongoDB Prod DB\nError: {err}")
        raise Exception("Fatal Error in MongoDB Boostrap")
    return



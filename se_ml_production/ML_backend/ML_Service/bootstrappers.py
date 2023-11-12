import os
import secret
from art import *
from bson import ObjectId
from pymongo import MongoClient
from google.cloud import storage

from spinner import Spinner # Import our custom spinner classes
from modelLoaders import load_bert_NER, load_bert_TOPIC

from geographyEntities import geography
from censusNeighborhood import neighborhood_mapping

def fetch_llm_models():
	"""
	Runs through google cloud bucket to fetch the approate LLM's for prediction
	"""
	try:
		storage_client = storage.Client()

		bucket = storage_client.bucket("naacp-models")
		blob = bucket.blob("BERTopic_Models/BERTopic_CPU_M1")
		blob.download_to_filename("./llm_models/BERTopic_CPU_M1")
	except Exception as e:
		print(f"Model fetching failed! {e}")
		raise Exception("Fatal Error in Fetching LLM models. Exiting...")
	return

def load_heirarch_data(db):
	"""
	Loads the Heirarchal data for mapping given after the topic model
	"""
	try:
		if (db == None):
			raise Exception("No database was given!")

		heirarch_db = db['Topic_Modeling_Pipeline_Data'].find_one(
			{"_id": ObjectId("654c38cff9689d923f1c3e9c")}
		)

		if (heirarch_db == None):
			raise Exception(f"heirarch_db Length: [{len(heirarch_db)}]")

		heirarch_data = heirarch_db["taxonomy_data"]
		return heirarch_data
	except Exception as err:
		print(f"Error loading hierarchal data!\nError: {err}")
		raise Exception("Fatal Error in fetching hierarchal data!")
	return


def bootstrap_pipeline():
	"""
	Bootstraps the entire application and makes the container ready to run for inferencing
	"""
	try:
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

		spinner = Spinner("Connecting to MongoDB...")
		spinner.start()
		client = MongoClient(secret.MONGO_URI_NAACP)
		db = client['ML_Data']
		spinner.stop()
		print()

		spinner = Spinner("Fetching Heirarchal Mappings...")
		spinner.start()
		heir_data = load_heirarch_data(db)
		spinner.stop()
		print()

		spinner = Spinner("Detecting & Making local directories...")
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
			if (not os.path.isfile("./llm_models/BERTopic_CPU_M1")):
				spinner.stop()
				spinner = Spinner("Model file not found! Pulling models...")
				spinner.start()
				fetch_llm_models()
				spinner.stop()
			else:
				spinner.stop()
                
		print("Model files successfully validated.\n")
		spinner = Spinner("Instantiating classes...")
		spinner.start()
		drop_geos = geography(
		    db=db
		)
		mappings = neighborhood_mapping(
		    db=db
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
		return year, dsource, dname, state, drop_geos, mappings, heir_data, saved_geocodes, nlp_ner, nlp_topic, db
	except Exception as e:
		spinner.err()
		print(f"Bootstrap Failed!!!\nFatal Error:{e}")
		raise Exception("Fatal Error in Bootstrapping ML Pipeline. Exiting...")
        
	return

def validate_bootstrap(year, dsource, dname, state, drop_geos, mappings, heir_data, saved_geocodes, nlp_ner, nlp_topic, db):
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
			"heir_data": heir_data,
			"saved_geocodes": saved_geocodes, 
			"nlp_ner": nlp_ner, 
			"nlp_topic": nlp_topic,
			"db": db
		}
        
		for var in variable_manifest.keys():
			if (variable_manifest[var] == None):
				raise Exception(f"{var} returned None!. Exiting...")
		spinner.stop()
		print()
		print("Validation Complete! Everything seems to be in order.")
	except Exception as e:
		spinner.err()
		print(f"Bootstrap Validation Failed!!!\nFatal Error:{e}")
		raise Exception("Fatal Error in Bootstrapping Validation. Exiting...")
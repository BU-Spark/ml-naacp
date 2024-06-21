import os
import json
import zipfile
from art import *
import pandas as pd
from bson import ObjectId
from pymongo import MongoClient
from google.cloud import storage

import secret
from Utils.spinner import Spinner
from Model_Utils.openAI import OpenAIClient
from Model_Utils.google_Maps import GoogleMapsClient
from Mongo_Utils.mongoDB_manager import MongoDBManager
from Model_Utils.model_Loaders import load_llama_7B, load_spanmarker_NER

def fetch_llm_models(zip_file=False, filename=None, google_repo_path=None):
	"""
	Runs through google cloud bucket to fetch the appropriate LLM's for prediction
	"""
	try:
		storage_client = storage.Client()
		bucket = storage_client.bucket("naacp-models")
		blob = bucket.blob(f"{google_repo_path}/{filename}")
		blob.download_to_filename(f"./llm_models/{filename}")
		
		if (zip_file):
			destination_file_name = f"./llm_models/{filename}"
			with zipfile.ZipFile(destination_file_name, 'r') as zip_ref: # Extract the ZIP
				zip_ref.extractall("./llm_models")
			os.remove(destination_file_name) # Remove the ZIP for cleanup
	except Exception as e:
		print(f"Model fetching failed! {e}")
		raise Exception("Fatal Error in Fetching LLM models. Exiting...")
	return

def load_taxonomy_lists(openAIClient):
	"""
	Loads the Taxonomy List based on heirarchies
	"""
	# Get embedding for ALL topics selected by BERTopic
	taxonomy_df = pd.read_csv('./data_prod/Content_Taxonomy.csv', skiprows=5, usecols=range(8))
	taxonomy_df.columns = taxonomy_df.iloc[0]
	taxonomy_df = taxonomy_df.tail(-1)

	tier_1_list = []
	tier_2_list = []
	tier_3_list = []
	tier_4_list = []
	for index, row in taxonomy_df.iterrows():
		if not pd.isnull(row['Tier 4']) and row['Tier 4'] != ' ':
			tier_1_label = row['Tier 1']
			tier_2_label = row['Tier 2']
			tier_3_label = row['Tier 3']
			tier_4_label = row['Tier 4']
			tier_4_list.append(f'{tier_1_label} - {tier_2_label} - {tier_3_label} - {tier_4_label}')
		elif not pd.isnull(row['Tier 3']) and row['Tier 3'] != ' ':
			tier_1_label = row['Tier 1']
			tier_2_label = row['Tier 2']
			tier_3_label = row['Tier 3']
			tier_3_list.append(f'{tier_1_label} - {tier_2_label} - {tier_3_label}')
		elif not pd.isnull(row['Tier 2']) and row['Tier 2'] != ' ':
			tier_1_label = row['Tier 1']
			tier_2_label = row['Tier 2']
			tier_2_list.append(f'{tier_1_label} - {tier_2_label}')
		else:
			tier_1_label = row['Tier 1']
			tier_1_list.append(f'{tier_1_label}')

	tier_1_list = list(set(tier_1_list))
	tier_2_list = list(set(tier_2_list))
	tier_3_list = list(set(tier_3_list))
	tier_4_list = list(set(tier_4_list))

	tier_1_embedding = [openAIClient.get_embedding(topic) for topic in tier_1_list]
	tier_2_embedding = [openAIClient.get_embedding(topic) for topic in tier_2_list]
	tier_3_embedding = [openAIClient.get_embedding(topic) for topic in tier_3_list]
	tier_4_embedding = [openAIClient.get_embedding(topic) for topic in tier_4_list]

	all_topics_list = []
	[all_topics_list.append(topic) for topic in tier_1_list]
	[all_topics_list.append(topic) for topic in tier_2_list]
	[all_topics_list.append(topic) for topic in tier_3_list]
	[all_topics_list.append(topic) for topic in tier_4_list]

	all_topics_embedding = []
	[all_topics_embedding.append(embedding) for embedding in tier_1_embedding]
	[all_topics_embedding.append(embedding) for embedding in tier_2_embedding]
	[all_topics_embedding.append(embedding) for embedding in tier_3_embedding]
	[all_topics_embedding.append(embedding) for embedding in tier_4_embedding]

	# Get embedding for the 230 topics selected by BERTopic 
	selected_taxonomy_df = pd.read_csv('./data_prod/embedding_similarity_label.csv')
	selected_taxonomy_df = selected_taxonomy_df.dropna(subset=['closest_topic'])
	selected_topics_list = selected_taxonomy_df['closest_topic'].values.tolist()

	selected_topics_embedding = [openAIClient.get_embedding(topic) for topic in selected_topics_list]

	# Alternative taxonomy: client's list of topics
	client_taxonomy_df = pd.read_excel('./data_prod/Asad_Topics_List.xlsx', names=['label'])
	client_taxonomy_df['ada_embedding'] = client_taxonomy_df['label'].map(openAIClient.get_embedding)

	return all_topics_embedding, selected_topics_embedding, client_taxonomy_df, all_topics_list, selected_topics_list

def fetch_and_load_taxonomy_lists(db, openAIClient, fetch=True):
	"""
	Fetches and Loads the Taxonomy Lists for Topic Modeling
	"""
	try:
		if (db == None):
			raise Exception("No Bucket Database Given!")
		if (fetch):
			bucket = db.bucket("ml_naacp_model_data")
			blob_1 = bucket.blob("Topic_Modeling_Pipeline_Data/Asad_Topics_List.xlsx")
			blob_1.download_to_filename("./data_prod/Asad_Topics_List.xlsx")
			blob_2 = bucket.blob("Topic_Modeling_Pipeline_Data/Content_Taxonomy.csv")
			blob_2.download_to_filename("./data_prod/Content_Taxonomy.csv")
			blob_3 = bucket.blob("Topic_Modeling_Pipeline_Data/embedding_similarity_label.csv")
			blob_3.download_to_filename("./data_prod/embedding_similarity_label.csv")
			blob_4 = bucket.blob("Entity_Recognition_Pipeline_Data/known_locs.json") # Here is for Explicit mention pass
			blob_4.download_to_filename("./data_prod/known_locs.json")
			all_topics_embedding, selected_topics_embedding, client_taxonomy_df, all_topics_list, selected_topics_list = load_taxonomy_lists(openAIClient)

			return all_topics_embedding, selected_topics_embedding, client_taxonomy_df, all_topics_list, selected_topics_list
	except Exception as err:
		print(f"Error fetching and loading the taxonomy data!\nError: {err}")
		raise Exception("Fatal Error in fetching taxonomy data!")
	return

def bootstrap_pipeline():
	"""
	Bootstraps the entire pipeline for inference
	"""
	try:
		file_manifest = [
			"known_locs.json", # Entity Recognition | Known Locs
			"Asad_Topics_List.xlsx", # Topic Modeling
			"Content_Taxonomy.csv", # Topic Modeling
			"embedding_similarity_label.csv", # Topic Modeling
		]
		dependency_resolver_arr = []
        
		tprint("BU Spark!", font="sub-zero")
		print("Bootstrapping pipeline... (This should only run once!)")
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

		spinner = Spinner("Connecting to Google Maps API...")
		spinner.start()
		googleMapsClient = GoogleMapsClient()
		googleMapsClient.createMapsClient(API_KEY=secret.GOOGLE_MAPS_KEY)
		spinner.stop()
		print()

		spinner = Spinner("Connecting to OpenAI...")
		spinner.start()
		openAIClient = OpenAIClient()
		openAIClient.createOpenAIClient(API_KEY=secret.OPENAI_KEY)
		spinner.stop()
		print()

		spinner = Spinner("Detecting & Making local directories for Models...")
		spinner.start()
		spinner.stop()
		llm_model_directory_path = "./llm_models"
		google_path_dir = "Llama_Models"
		model_file = "llama-2-7b-chat.Q4_K_M.gguf.zip"
		required_model_file = "llama-2-7b-chat.Q4_K_M.gguf"
		if (not os.path.exists(llm_model_directory_path)):
			print(f"No {llm_model_directory_path}! Creating...")
			os.makedirs(llm_model_directory_path)
			spinner = Spinner("Fetching models (This might take a while)...")
			spinner.start()
			fetch_llm_models(zip_file=True, filename=model_file, google_repo_path=google_path_dir)
			spinner.stop()
		else:
			print(f"Found! {llm_model_directory_path}!")
			spinner = Spinner("Validating model files...")
			spinner.start()
			if (not os.path.isfile(f"./llm_models/{required_model_file}")):
				spinner.stop()
				spinner = Spinner("Model file not found! Pulling models...")
				spinner.start()
				fetch_llm_models(zip_file=True, filename=model_file, google_repo_path=google_path_dir)
				spinner.stop()
			else:
				spinner.stop()
		print("Model files successfully validated.\n")

		spinner = Spinner("Detecting & making local directory for required model files...")
		spinner.start()
		spinner.stop()
		geodata_directory_path = "./data_prod"
		if (not os.path.exists(geodata_directory_path)):
			print(f"No {geodata_directory_path}! Creating...")
			os.makedirs(geodata_directory_path)
			dependency_resolver_arr = len(file_manifest) * [True]
		else:
			print(f"Found! {geodata_directory_path}!")
			spinner = Spinner("Validating geodata files...")
			spinner.start()
			for file in file_manifest:
				if (not os.path.isfile(f"./data_prod/{file}")): # We flip the truth values, if the file is not found, we print True
					dependency_resolver_arr.append(True)
				else:
					dependency_resolver_arr.append(False)
			spinner.stop()
		print("Geodata files successfully validated and updated dependency array.\n")

		spinner = Spinner("Fetching Taxonomy Mappings using OpenAI Embedding Model (This might take a while)...")
		spinner.start()
		all_topics_embedding, selected_topics_embedding, client_taxonomy_df, all_topics_list, selected_topics_list = fetch_and_load_taxonomy_lists(db, openAIClient, fetch=True) #fetch=any(dependency_resolver_arr)
		spinner.stop()
		print()
        
		spinner = Spinner("Loading Models...")
		spinner.start()
		nlp_llm = load_llama_7B()
		nlp_ner = load_spanmarker_NER()
		spinner.stop()
        
		print("\nBootstrap complete!\n")
		return all_topics_embedding, selected_topics_embedding, client_taxonomy_df, all_topics_list, selected_topics_list, openAIClient, googleMapsClient, nlp_llm, nlp_ner, db, db_manager
	except Exception as e:
		spinner.err()
		print(f"Bootstrap Failed!!!\nFatal Error:{e}")
		raise Exception("Fatal Error in Bootstrapping ML Pipeline. Exiting...")
        
	return

def validate_bootstrap(all_topics_embedding, selected_topics_embedding, client_taxonomy_df, all_topics_list, selected_topics_list, openAIClient, googleMapsClient, nlp_llm, nlp_ner, db, db_manager):
	"""
	Validates the bootstrap variables, checking if they exist.
	"""
	try:
		spinner = Spinner("Validating Bootstrap variables...")
		spinner.start()

		variable_manifest = {
			"all_topics_embedding": all_topics_embedding, 
			"selected_topics_embedding": selected_topics_embedding, 
			"client_taxonomy_df": client_taxonomy_df,
			"all_topics_list": all_topics_list,
			"selected_topics_list": selected_topics_list,
			"openAIClient": openAIClient,
			"googleMapsClient": googleMapsClient,
			"nlp_llm": nlp_llm,
			"nlp_ner": nlp_ner, 
			"db": db,
			"db_manager": db_manager
		}
        
		for var in variable_manifest.keys():
			if (variable_manifest[var] is None):
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

        db_prod = client[secret.db_name]
            
        # Here we check for the upload collection and make it if it doesn't exist
        collection_list = db_prod.list_collection_names()
        for collection in defined_collection_names:
            if collection not in collection_list:
                db_prod.create_collection(collection)
                print(f"[INFO] Collection '{collection}' created.\n")
        spinner.stop()
    except Exception as err:
        spinner.err()
        print(f"[Error!] Error in Bootstrapping MongoDB Prod DB\nError: {err}")
        raise Exception("Fatal Error in MongoDB Boostrap")
    return



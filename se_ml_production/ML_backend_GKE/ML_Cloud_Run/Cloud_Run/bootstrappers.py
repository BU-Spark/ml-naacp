import os
import json
import zipfile
import secret
from art import *
from bson import ObjectId
from pymongo import MongoClient
import sys

from google.cloud import storage
from Utils.spinner import Spinner
from Mongo_Utils.mongoDB_manager import MongoDBManager

def bootstrap_pipeline():
	"""
	Bootstraps the entire pipeline for inference
	"""
	try:
		tprint("BU Spark!", font="sub-zero")
		print("Spinning up Container (Bootstrapping)")
		print()

		spinner = Spinner("Creating MongoDB Manager..")
		spinner.start()
		db_manager = MongoDBManager()
		spinner.stop()
		print()

		spinner = Spinner("Connecting to Google Cloud Storage Bucket...")
		spinner.start()
		gcp_db = storage.Client()
		spinner.stop()
		print()
        
		print("\nBootstrap complete!\n")
		return db_manager, gcp_db
	except Exception as e:
		spinner.err()
		print(f"Bootstrap Failed!!!\nFatal Error:{e}")
		raise Exception("Fatal Error in Bootstrapping ML Pipeline [Cloud Run]. Exiting...")
        
	return

def validate_bootstrap(db_manager, gcp_db):
	"""
	Validates the bootstrap variables, checking if they exist.
	"""
	try:
		spinner = Spinner("Validating Bootstrap variables...")
		spinner.start()

		variable_manifest = {
			"db_manager": db_manager,
			"gcp_db": gcp_db
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

        db_prod = client[secret.db_name]
            
        # Here we check for the upload collection and make it if it doesn't exist
        collection_list = db_prod.list_collection_names()
        for collection in defined_collection_names:
            if collection not in collection_list:
                db_prod.create_collection(collection)
                print(f"[INFO] Collection '{collection}' created.")
        spinner.stop()
    except Exception as err:
        spinner.err()
        print(f"[Error!] Error in Bootstrapping MongoDB Prod DB\nError: {err}")
        raise Exception("Fatal Error in MongoDB Boostrap")
    return



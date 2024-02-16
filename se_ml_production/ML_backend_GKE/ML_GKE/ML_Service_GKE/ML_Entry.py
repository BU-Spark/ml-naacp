import os
import sys
import shutil
import inspect
import uuid # For upload ID unique generation
import pandas as pd
from typing import Union
from datetime import datetime
from urllib.parse import unquote_plus
from fastapi import APIRouter

from global_state import global_instance
from csv_funcs import read_csv, validate_csv
from ML_Pred_Funcs.ML_funcs import process_data, topic_modeling
from Mongo_Utils.production_mongo_funcs import send_to_production, send_Discarded
from Mongo_Utils.mongo_funcs import update_job_status, connect_MongoDB_Prod


# Data packing scheme, function attributes must be len(data_schema) + 1
# Ideally, the developer should provide a function to figure out how to pack the data based on their needs
def package_data_to_dict(
	data_schema, 
	id,
	neighborhoods,
	position_section,
	tracts,
	author,
	body,
	content_id,
	hl1,
	hl2,
	pub_date,
	pub_name,
	link,
	method,
	ent_geocodes
):
	try:
		args, _, _, values = inspect.getargvalues(inspect.currentframe())
		args.pop(0) # we pop off data_schema
		for arg in args:
			data_schema[arg].append(values[arg])
	except KeyError as ke:
		print("Key not found in data schema!")
		print(f"Raw Error: {ke}")

	return data_schema

# ====== Here we run our pipeline ====== 
def run_pipeline(df, upload_id: str, user_id: str, upload_timestamp: str):
	db_manager = global_instance.get_data("db_manager")

	try:
    	# To run the pipeline, two things we need to have defined is the data_schmea and data packing func
		data_schema = {
			"id": [],
			"neighborhoods": [],
			"position_section": [],
			"tracts": [],
			"author": [],
			"body": [],
			"content_id": [],
			"hl1": [],
			"hl2": [],
			"pub_date": [],
			"pub_name": [],
			"link": [],
			"method": [],
			"ent_geocodes": []
		}

		# Here we need to generate an Upload ID & have the user ID ready
		# Assuming this runs sequentially, these variables shouldn't be changed until the prediction is finished!
		global_instance.update_data("upload_id", upload_id) # Should only run once!
		global_instance.update_data("userID", user_id)
		global_instance.update_data("upload_timestamp", upload_timestamp)
		global_instance.update_data("upload_status", "PROCESSING")

		# We are now in the processing state!
		db_manager.run_job(
			update_job_status, 
			db_manager.act_con[0]['connection'], # Argument 1 (1st connection)
			global_instance.get_data("upload_id"), 
			global_instance.get_data("userID"),
			global_instance.get_data("upload_timestamp"), 
			df.shape[0], 
			global_instance.get_data("upload_status"),
			"INFERENCE PIPELINE IS PROCESSING.",
			connection_obj=db_manager.act_con[0]
		)

		# Conduct Entity Recognition and return the new df
		print("[INFO] Processing through Entity Recongition.")
		entity_rec_df, discarded_articles = process_data(list(range(df.shape[0])), df, data_schema, package_data_to_dict, global_instance.get_data("nlp_ner"))
		print("Discarded Articles COUNT", len(discarded_articles))

		if (entity_rec_df.empty):
			print("[INFO] Entity recongition came up with no locations! Aborting...")
			global_instance.update_data("upload_status", "NO LOCATIONS")
			db_manager.run_job(
				update_job_status, 
				db_manager.act_con[0]['connection'], # Argument 1 (1st connection)
				global_instance.get_data("upload_id"), 
				global_instance.get_data("userID"),
				global_instance.get_data("upload_timestamp"), 
				0, 
				global_instance.get_data("upload_status"),
				"NO LOCATIONS FOUND.",
				connection_obj=db_manager.act_con[0]
			)
			db_manager.run_job(
				send_Discarded,
				db_manager.act_con[0]['connection'], # Argument 1 (1st connection)
				discarded_articles,
				connection_obj=db_manager.act_con[0]
			)
			return

		print("[INFO] Processing through Topic Modeling.")
		final_df = topic_modeling(entity_rec_df)
		
		# Here we just add the UserID and UploadID
		final_df["userID"] = global_instance.get_data("userID")
		final_df["uploadID"] = global_instance.get_data("upload_id")

		print("[DEBUG] Final DF.")
		print(final_df)

		print("FINAL DF ARTICLE COUNT:", final_df)

		# Ideally, we should send final_df to a data warehouse

		# Prune the uneeded columns for production
		packaged_data_df = final_df.drop(columns=[
		    'method',
		    'ent_geocodes',
		    'bertopic_topic_label'
		])

		print("[INFO] Sending Inferences to Production DB.")
		db_manager.run_job(
			send_to_production, # Send the data to MongoDB
			db_manager.act_con[0]['connection'], # Argument 1 (1st connection)
			packaged_data_df,
			connection_obj=db_manager.act_con[0]
		)
		db_manager.run_job(
			send_Discarded, # Send the discarded articles to avoid duplicates
			db_manager.act_con[0]['connection'], # Argument 1 (1st connection)
			discarded_articles,
			connection_obj=db_manager.act_con[0]
		)

		## Here we mark the job as complete
		global_instance.update_data("upload_status", "SUCCESS")

		db_manager.run_job(
			update_job_status, 
			db_manager.act_con[0]['connection'], # Argument 1 (1st connection)
			global_instance.get_data("upload_id"), 
			global_instance.get_data("userID"),
			global_instance.get_data("upload_timestamp"), 
			entity_rec_df.shape[0], 
			global_instance.get_data("upload_status"),
			"COMPLETE",
			connection_obj=db_manager.act_con[0]
		)

		#db_manager.force_close_connection(unique_id=db_manager.act_con[0]['id']) # Close the connection as pymongo in GCP doesnt close it

		print(db_manager) # Check MongoDB Statuses

		return
	except Exception as e:
		global_instance.update_data("upload_status", "FAILED")

		db_manager.run_job(
			update_job_status, 
			db_manager.act_con[0]['connection'], # Argument 1 (1st connection)
			global_instance.get_data("upload_id"), 
			global_instance.get_data("userID"),
			global_instance.get_data("upload_timestamp"), 
			-1, 
			global_instance.get_data("upload_status"),
			f"{e}",
			connection_obj=db_manager.act_con[0]
		)
		print(db_manager) # Check MongoDB Statuses
		raise
	return





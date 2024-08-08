import pandas as pd

from global_state import global_instance
from ML_Pred_Funcs.ML_funcs import geolocate_articles, topic_modeling
from Mongo_Utils.production_mongo_funcs import send_to_production
from Mongo_Utils.mongo_funcs import update_job_status

# ====== Here we run our pipeline ====== 
def run_pipeline(df, upload_id: str, user_id: str, upload_timestamp: str):
	db_manager = global_instance.get_data("db_manager")

	try:
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
		print("[INFO] Processing through Geolocation pipeline.")
		processing_df = geolocate_articles(df)

		if (processing_df.empty):
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
			return

		print("[INFO] Processing through Topic Modeling.")
		final_df = topic_modeling(processing_df)
		
		# Here we just add the UserID and UploadID
		final_df["userID"] = global_instance.get_data("userID")
		final_df["uploadID"] = global_instance.get_data("upload_id")

		final_df = final_df.drop(columns=[
			"content_id",
			"Body",
			"Headline"
		])
		final_df = pd.concat([final_df, df], axis=1)

		print("[DEBUG] Final DF.")
		print(final_df)

		packaged_data_df = final_df.drop(columns=[
		    'Explicit_Pass',
		    'NER_Pass',
			'LLM_Pass',
			'topic_model_body',
			'tokens',
			'ada_embedding',
			'closest_topic_all',
			'closest_topic_selected',
		])

		"""
		Index(['content_id', 'Headline', 'Body', 'Tracts', 'closest_topic_client',
       'userID', 'uploadID', 'Byline', 'Body', 'Headline', 'Publish Date',
       'Publisher', 'Paths', 'content_id'],
		"""

		packaged_data_df = packaged_data_df.rename(columns={
			"Byline": "author",
			"Body": "body",
			"Headline": "hl1",
			"Publish Date": "pub_date",
			"Publisher": "pub_name",
			"Paths": "link",
			"Tracts": "tracts",
			"Counties": "counties",
			"Neighborhoods": "neighborhoods",
			"Locations": "locations",
			"closest_topic_client": "openai_labels",
		})
		print(f"[DEBUG] Dataframe tracts {packaged_data_df["tracts"]}")
		print(f"[DEBUG] Dataframe Columns {packaged_data_df.columns}")

		print("[INFO] Sending Inferences to Production DB.")
		db_manager.run_job(
			send_to_production, # Send the data to MongoDB
			db_manager.act_con[0]['connection'], # Argument 1 (1st connection)
			packaged_data_df,
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
			processing_df.shape[0], 
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





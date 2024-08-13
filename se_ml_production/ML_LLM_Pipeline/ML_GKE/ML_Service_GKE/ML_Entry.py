import pandas as pd

from global_state import global_instance
from ML_Pred_Funcs.ML_funcs import geolocate_articles, topic_modeling
from Mongo_Utils.upload_Data import send_to_production
from Mongo_Utils.mongo_funcs import update_job_status

def update_job(size, message, status="PROCESSING"):
	"""
	Update the job status in the database
	"""
	db_manager = global_instance.get_data("db_manager")

	db_manager.run_job(
		update_job_status, 
		db_manager.act_con[0]['connection'], # Argument 1 (1st connection)
		global_instance.get_data("upload_id"), 
		global_instance.get_data("userID"),
		global_instance.get_data("upload_timestamp"), 
		size, 
		status,
		message,
		connection_obj=db_manager.act_con[0]
	)

	return

def format_df(df, articles):
	# Here we just add the UserID and UploadID
	df["userID"] = global_instance.get_data("userID")
	df["uploadID"] = global_instance.get_data("upload_id")

	df = df.drop(columns=["content_id", "Body", "Headline"])

	final_df = pd.concat([df, articles], axis=1)

	print("[DEBUG] Final DF.")
	print(final_df)

	print(f"[DEBUG] Number of articles located per Pass: ")
	passes = ["Explicit_Pass", "NER_Pass", "LLM_Pass"]

	for column in passes:
		count = final_df[column].notna().sum()
		print(f"{column} located {count} articles.")
		
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

	packaged_data_df = packaged_data_df.rename(columns={
		"Byline": "author",
		"Body": "body",
		"Headline": "hl1",
		"Publish Date": "pub_date",
		"Publisher": "pub_name",
		"Paths": "link",
		"Tracts": "tracts",
		"Coordinates": "coordinates",
		"Counties": "counties",
		"Neighborhoods": "neighborhoods",
		"Locations": "locations",
		"closest_topic_client": "openai_labels",
	})

	print(f"[DEBUG] Dataframe Columns {packaged_data_df.columns}")
	return packaged_data_df

def send_data(df):
	"""
	Send the data to the MongoDB
	"""
	db_manager = global_instance.get_data("db_manager")

	# Save the data to the MongoDB
	db_manager.run_job(
		send_to_production, # Send the data to MongoDB
		db_manager.act_con[0]['connection'], # Argument 1 (1st connection)
		df,
		connection_obj=db_manager.act_con[0]
	)

	return

# ====== Here we run our pipeline ====== 
def run_pipeline(df, upload_id: str, user_id: str, upload_timestamp: str):
	db_manager = global_instance.get_data("db_manager")

	# Here we need to generate an Upload ID & have the user ID ready
	# Assuming this runs sequentially, these variables shouldn't be changed until the prediction is finished!
	global_instance.update_data("upload_id", upload_id) # Should only run once!
	global_instance.update_data("userID", user_id)
	global_instance.update_data("upload_timestamp", upload_timestamp)
	global_instance.update_data("upload_status", "PROCESSING")

	try:
		update_job(df.shape[0], "INFERENCE PIPELINE IS PROCESSING.")
		
		# We are now in the processing state! Process articles in batches of 100
		batch_size = 100
		batch_count = 1 + df.shape[0] // batch_size
		total_count = 0
		for batch in range(0, df.shape[0], batch_size):
			articles = df[batch:batch+batch_size]

			print(f"[INFO] Processing Batch {batch}/{batch_count}.")
			update_job(articles.shape[0], f"INFERENCE PIPELINE IS PROCESSING [{batch + 1}/{batch_count}].")

			# Conduct Entity Recognition and return the new df
			print("[INFO] Processing through Geolocation pipeline.")
			processing_df = geolocate_articles(articles)

			if (processing_df.empty):
				print("[INFO] Entity recongition came up with no locations! Skipping to next batch.")
				update_job(0, f"NO LOCATIONS FOUND for [{batch}/{batch_count}].", "NO LOCATIONS")
				continue

			print("[INFO] Processing through Topic Modeling.")
			final_df = topic_modeling(processing_df)

			print("[INFO] Formatting Data for MongoDB.")
			packaged_data_df = format_df(final_df, articles)

			print("[INFO] Sending Inferences to Production DB.")
			send_data(packaged_data_df)
			
			print(f"[INFO] Batch Complete! Recognized {packaged_data_df.shape[0]} articles.")
			total_count += packaged_data_df.shape[0]

		print("[INFO] Inference Pipeline Complete!")
		global_instance.update_data("upload_status", "SUCCESS")
		update_job(total_count, "INFERENCE PIPELINE IS COMPLETE.", "SUCCESS")

		#db_manager.force_close_connection(unique_id=db_manager.act_con[0]['id']) # Close the connection as pymongo in GCP doesnt close it

		print(db_manager) # Check MongoDB Statuses

		return
	except Exception as e:
		global_instance.update_data("upload_status", "FAILED")

		update_job(-1, f"INFERENCE PIPELINE FAILED: {e}.", "FAILED")

		print(db_manager) # Check MongoDB Statuses
		raise





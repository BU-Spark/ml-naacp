import pandas as pd

import secret

from global_state import global_instance
from ML_Pred_Funcs.ML_funcs import geolocate_articles, topic_modeling
from Mongo_Utils.upload_Data import send_to_production
from Mongo_Utils.mongo_funcs import update_job_status

from Mongo_Utils.mongo_neighborhoods import get_neighborhoods

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

	final_df = final_df.dropna(subset=["locations"]).reset_index(drop=True)
	
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
		"closest_topic_client": "openai_labels",
	})

	print(f"[DEBUG] Dataframe Columns {packaged_data_df.columns}")
	return packaged_data_df

def is_duplicate_article(tag, articles_collection):
	query = {'_id': tag}
	return articles_collection.find_one(query) is not None

def run_validation(client, df):
	db_prod = client[secret.db_name]
	collection_list = db_prod.list_collection_names()
	org_Id = global_instance.get_data("orgID")
	collection_name = f"articles_data_{org_Id}"

	if (collection_name in collection_list):
		articles_collection = db_prod[collection_name]
		df['is_duplicate'] = df['content_id'].apply(lambda tag: is_duplicate_article(tag, articles_collection))
		print(df[df['is_duplicate'] == False]['content_id'])
		df = df.drop(df[df['is_duplicate']].index).drop(columns='is_duplicate')
	return df

def partial_df(df):
	partial_df = df.copy()

    # Fix path link
	def fix_link(link):
		if link is None:
			return None
		if link.startswith('http'):
			return link
		if link.endswith(" (Permalink)"):
			return "https://www.wgbh.org" + link[:-11]
		else:
			return link
    
	partial_df['Paths'] = partial_df['Paths'].apply(fix_link)

	# Drop rows where at least one of the specified columns is empty
	columns_to_check = ['Headline', 'Body', 'Publisher', 'Publish Date', 'Byline', 'Paths']
	partial_df = partial_df.dropna(subset=columns_to_check).reset_index(drop=True)

	if len(partial_df) == 0:
		return None

	# Drop empty rows too
	partial_df = partial_df[~partial_df['Body'].apply(lambda x: isinstance(x, float))]
	partial_df = partial_df[~partial_df['Headline'].apply(lambda x: isinstance(x, float))]

	return partial_df

def send_data(df, org_key):
	"""
	Send the data to the MongoDB
	"""
	db_manager = global_instance.get_data("db_manager")

	# Save the data to the MongoDB
	db_manager.run_job(
		send_to_production, # Send the data to MongoDB
		db_manager.act_con[0]['connection'], # Argument 1 (1st connection)
		df,
		org_key,
		connection_obj=db_manager.act_con[0]
	)

	return

# ====== Here we run our pipeline ====== 
def run_pipeline(df, upload_id: str, user_id: str, org_id: str, upload_timestamp: str):
	db_manager = global_instance.get_data("db_manager")

	# Here we need to generate an Upload ID & have the user ID ready
	# Assuming this runs sequentially, these variables shouldn't be changed until the prediction is finished!
	global_instance.update_data("upload_id", upload_id) # Should only run once!
	global_instance.update_data("userID", user_id)
	global_instance.update_data("orgID", org_id) 
	global_instance.update_data("upload_timestamp", upload_timestamp)
	global_instance.update_data("upload_status", "PROCESSING")
	
	org_key = org_id

	# Retrieve the relevant neighborhood and tract data
	tract_map, neigh_map = db_manager.run_job(
            get_neighborhoods,
            db_manager.act_con[0]['connection'], # Argument 1 (1st connection)
			org_key,
            connection_obj=db_manager.act_con[0]
        )
	global_instance.update_data("neigh_map", neigh_map)
	global_instance.update_data("tract_map", tract_map)

	try:
		# Remove duplicates
		df = run_validation(db_manager.act_con[0]['connection'], df)
		if (df.empty):
			global_instance.update_data("upload_status", "ALL DUPLICATES")
			update_job(0, "ALL DUPLICATES FOUND.", "SUCCESS")
			return
		
		update_job(df.shape[0], "INFERENCE PIPELINE IS PROCESSING.")
		
		# We are now in the processing state! Process articles in batches

		# Self adjust the batch size 
		article_count = df.shape[0]
		# Idk if it affects performance significantly. Other than for the initial batch of new orgs,
		# continuous uploads would be much smaller, so it should be fine either way.
		if article_count < 10:
			batch_size = 1
		elif article_count < 200: 
			batch_size = 10
		else:
			batch_size = 100
		batch_count = 1
		batch_total = 1 + df.shape[0] // batch_size
		total_count = 0
		for batch in range(0, df.shape[0], batch_size):
			articles = df[batch:batch+batch_size]

			print(f"[INFO] Processing Batch {batch_count} of {batch_total}.")
			update_job(articles.shape[0], f"INFERENCE PIPELINE IS PROCESSING [{batch}/{article_count}].")

			# Cleaning the articles
			partial = partial_df(articles)
			if partial is None or len(partial) == 0:
				print(f"[WARNING] No articles passed the cleaning pipeline")
				batch_count += 1
				continue
				
			# Conduct Entity Recognition and return the new df
			print("[INFO] Processing through Geolocation pipeline.")
			processing_df = geolocate_articles(partial)

			if (processing_df.empty):
				print("[INFO] Entity recongition came up with no locations! Skipping to next batch.")
				update_job(0, f"NO LOCATIONS FOUND for [{batch}/{article_count}].", "PROCESSING")
				continue

			print("[INFO] Processing through Topic Modeling.")
			final_df = topic_modeling(processing_df)

			print("[INFO] Formatting Data for MongoDB.")
			packaged_data_df = format_df(final_df, partial)

			print("[INFO] Sending Inferences to Production DB.")
			send_data(packaged_data_df, org_key)
			
			print(f"[INFO] Batch Complete! Recognized {packaged_data_df.shape[0]} articles.")
			total_count += packaged_data_df.shape[0]
			batch_count += 1

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





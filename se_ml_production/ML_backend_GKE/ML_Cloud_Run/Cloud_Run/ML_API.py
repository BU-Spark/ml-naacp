import os
import sys
import shutil
import inspect
import uuid # For upload ID unique generation
import pandas as pd
from typing import Union
from datetime import datetime
from fastapi.responses import JSONResponse
from fastapi import UploadFile, HTTPException, Query, Body, Form
from urllib.parse import unquote_plus
from fastapi import APIRouter

from io import StringIO
from google.cloud import storage
from global_state import global_instance
from csv_funcs import read_csv, validate_csv
from Mongo_Utils.mongo_funcs import update_job_status

import json
from typing import Callable
from concurrent import futures
from google.cloud import pubsub_v1
import hashlib

ml_router = APIRouter()

def get_callback(
	publish_future: pubsub_v1.publisher.futures.Future, data: str
) -> Callable[[pubsub_v1.publisher.futures.Future], None]:
	def callback(publish_future: pubsub_v1.publisher.futures.Future) -> None:
		try:
			# Wait 60 seconds for the publish call to succeed.
			print(publish_future.result(timeout=60))
		except futures.TimeoutError:
			print(f"Publishing {data} timed out.")

	return callback

def upload_df_to_gcs(gcp_db, bucket_name, destination_blob_name, df):
	try:
		storage_client = gcp_db
		bucket = storage_client.bucket(bucket_name)
		blob = bucket.blob(destination_blob_name)

		csv_buffer = StringIO()
		df.to_csv(csv_buffer, index=False)

		blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
	except Exception as e: 
		raise Exception (f"Error in uploading to GCP: {e}")
	return

def create_content_ids(df):
	df['content_id'] = df['Body'].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
	return

### API Endpoints
@ml_router.post("/upload_csv")
async def upload_file(file: UploadFile = None, user_id: str = Form(...)):
	db_manager = global_instance.get_data("db_manager")
	gcp_db = global_instance.get_data("gcp_db")

	try:
		# Here we need to generate an Upload ID & have the user ID ready
		# Assuming this runs sequentially, these variables shouldn't be changed until the prediction is finished!
		global_instance.update_data("upload_id", str(uuid.uuid4())) # Should only run once!
		global_instance.update_data("userID", user_id)
		global_instance.update_data("upload_timestamp", datetime.now())
		global_instance.update_data("upload_status", "VALIDATING")

		db_manager.run_job(
			update_job_status, 
			db_manager.act_con[0]['connection'], # Argument 1 (1st connection)
			global_instance.get_data("upload_id"), 
			global_instance.get_data("userID"),
			global_instance.get_data("upload_timestamp"), 
			-1, 
			global_instance.get_data("upload_status"),
			"JOB IS VALIDATING",
			connection_obj=db_manager.act_con[0]
		)

		df = await read_csv(file)

		print(f"[DEBUG] Recieved Columns:", df.columns) # For debug stuff
		print("[INFO] Checking for duplicates!")
		cleaned_df = validate_csv(df) 
		print(f"[DEBUG] Cleaned DF\n{cleaned_df}") # For debug stuff

		if (cleaned_df.empty):
			print("[INFO] Given DF is all duplicates")
			global_instance.update_data("upload_status", "ALL DUPLICATES")

			db_manager.run_job(
				update_job_status, 
				db_manager.act_con[0]['connection'], # Argument 1 (1st connection)
				global_instance.get_data("upload_id"), 
				global_instance.get_data("userID"),
				global_instance.get_data("upload_timestamp"), 
				0, 
				global_instance.get_data("upload_status"),
				"ALL DUPLICATES",
				connection_obj=db_manager.act_con[0]
			)

			return JSONResponse(content={"filename": file.filename, "status": "All are DUPLICATES. No files processed."}, status_code=200)
		
		create_content_ids(cleaned_df)

		# If there are no duplicates, we convert the pd -> csv then we upload to google storage bucket
		upload_df_to_gcs(gcp_db, "shiply_csv_bucket", str(
			"data/" + global_instance.get_data("upload_id") + "-" + global_instance.get_data("userID") + ".csv"
			), cleaned_df)

		# We then publish the information with all the meta info
		project_id = "special-michelle"
		topic_id = "shiply_upload_csv"

		publisher = pubsub_v1.PublisherClient()
		topic_path = publisher.topic_path(project_id, topic_id)
		publish_futures = []

		data = {
			"upload_id": global_instance.get_data("upload_id"),
			"userID": global_instance.get_data("userID"),
			"uploadTimeStamp": str(global_instance.get_data("upload_timestamp"))
		}
		data_str = json.dumps(data)
		publish_future = publisher.publish(topic_path, data_str.encode("utf-8"))
		publish_future.add_done_callback(get_callback(publish_future, data)) # Non-blocking. Publish failures are handled in the callback function.
		publish_futures.append(publish_future)

		# Wait for all the publish futures to resolve before exiting.
		futures.wait(publish_futures, return_when=futures.ALL_COMPLETED)

		print(f"[INFO]Published messages with error handler to {topic_path}.")

		# Signal that the published message went through
		global_instance.update_data("upload_status", "PUBLISHED MESSAGE")

		db_manager.run_job(
			update_job_status, 
			db_manager.act_con[0]['connection'], # Argument 1 (1st connection)
			global_instance.get_data("upload_id"), 
			global_instance.get_data("userID"),
			global_instance.get_data("upload_timestamp"), 
			-1, 
			global_instance.get_data("upload_status"),
			"MESSAGE PUBLISHED. JOB IS ENQUEUED",
			connection_obj=db_manager.act_con[0]
		)

		#db_manager.force_close_connection(unique_id=db_manager.act_con[0]['id']) # Close the connection as pymongo in GCP doesnt close it

		print(db_manager) # Check MongoDB Statuses

		return JSONResponse(content={"filename": file.filename, "status": "file uploaded"}, status_code=200)
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

		raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# Possibly Decprecated
@ml_router.post("/upload_RSS")
async def upload_RSS(RSS_Link: str = Body(..., description="The RSS feed link")):
	try:
		decoded_url = unquote_plus(RSS_Link)
		print("RSS Link:", decoded_url)
		return JSONResponse(content={"RSS Link": decoded_url, "status": "RSS Link Successfully Uploaded"}, status_code=200)
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

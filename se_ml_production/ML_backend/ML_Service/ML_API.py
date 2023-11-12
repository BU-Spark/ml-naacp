import os
import sys
import shutil
import inspect
import pandas as pd
from typing import Union
from fastapi.responses import JSONResponse
from fastapi import UploadFile, HTTPException, Query, Body
from urllib.parse import unquote_plus
from fastapi import APIRouter
from main import app # We need to import all the state variables

from csv_funcs import read_csv, validate_csv
from ML_funcs import process_data

ml_router = APIRouter()

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

### API Endpoints
@ml_router.post("/upload_csv")
async def upload_file(file: UploadFile = None):
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

		df = await read_csv(file)

		print(df.columns) # For debug stuff
		validate_csv(df) 
		print(df) # For debug stuff

		# Conduct Entity Recognition and return the new df
		entity_rec_df = process_data(list(range(1500, 1550)), df, data_schema, package_data_to_dict, app.state.nlp_ner)

		print()
		print("New DF For entity Recognition")
		print(entity_rec_df)

		return JSONResponse(content={"filename": file.filename, "status": "file uploaded"}, status_code=200)
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@ml_router.post("/upload_RSS")
async def upload_RSS(RSS_Link: str = Body(..., description="The RSS feed link")):
	try:
		decoded_url = unquote_plus(RSS_Link)
		print("RSS Link:", decoded_url)
		return JSONResponse(content={"RSS Link": decoded_url, "status": "RSS Link Successfully Uploaded"}, status_code=200)
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")





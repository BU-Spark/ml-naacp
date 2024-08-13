import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import secret

from Mongo_Utils.mongo_neighborhoods import neigh_tract_dict
from Mongo_Utils.mongo_funcs import get_collection, convert_to_datesum

# ==== Packing Funcs ====
def send_to_production(client, df):
	try:
		db_prod = client[secret.db_name]

		# Pack and send all articles
		pack_articles(db_prod, df)
		pack_neighborhoods(db_prod, df)
		pack_topics(db_prod, df)
		pack_tracts(db_prod, df)
		pack_locations(db_prod, df)

	except Exception as err:
		print(f"[Error!] Error in sending data to MongoDB Prod DB\nError: {err}")
		raise Exception("Fatal Error in sending to production")
	return

def pack_articles(db_prod, df):
	try:
		df['dateSum'] = df['pub_date'].apply(convert_to_datesum)

		article_payload = df.to_dict(orient='records')

		articles_collection = get_collection(db_prod, "articles_data")
		articles_collection.insert_many(article_payload)

		print("[INFO] Articles Successfully inserted!")
		return
	except Exception as err:
		raise Exception(f"[Error!] Error in sending Article Data\nError: {err}")
	return

def pack_neighborhoods(db_prod, df):
	try:
		neigh_collection = get_collection(db_prod, "neighborhood_data")

		# TODO: Delete this after neighborhood data is updated
		# for neighborhood in neigh_tract_dict.keys():
		# 	neigh_collection.update_one(
		# 		{'value': neighborhood},
		# 		{'$setOnInsert': {'tracts': neigh_tract_dict[neighborhood]}},
		# 		upsert = True # Creates a new document of it if it doesn't exist
		# 	)
		# print("[INFO] Neighborhoods Collection Successfully Populated!")

		# Save all new neighborhoods with associated tracts and articles
		for n, neighborhoods in enumerate(df['neighborhoods']):
			for neighborhood in neighborhoods:
				neigh_collection.update_one(
					{'value': neighborhood},
					{'$addToSet': {'articles': df['content_id'][n]}}
				)   

		print("[INFO] Neighborhoods Successfully inserted!")
	except Exception as err:
		raise Exception(f"[Error!] Error in sending Neighborhood Data\nError: {err}")
	return

def pack_topics(db_prod, df):
	try:
		topic_collection = get_collection(db_prod, "topics_data")

		# Save all new topics with associated articles
		for n, topic in enumerate(df["openai_labels"]):
			topic_collection.update_one(
				{'value': topic},
				{'$addToSet': {'articles': df["content_id"][n]}},
				upsert = True 
			)          
		print("[INFO] Topics Successfully inserted!")
	except Exception as err:
		raise Exception(f"[Error!] Error in sending Topics Data\nError: {err}")
	return

def pack_tracts(db_prod, df):
	try:
		tract_collection = get_collection(db_prod, "tracts_data")

		# Save all new tracts with associated articles and neighborhoods
		for n, tracts in enumerate(df['tracts']):
			for i, tract in enumerate(tracts):
				tract_collection.update_one(
					{'tract': tract},
					{'$addToSet': {'articles': df['content_id'][n]},
					 '$setOnInsert': {'neighborhood': df["neighborhoods"][n][i]}
					}, upsert=True
    			)
			  
		print("[INFO] Tracts Successfully inserted!")
	except Exception as err:
		raise Exception(f"[Error!] Error in sending Tracts Data\nError: {err}")
	return

def pack_locations(db_prod, df):
	try:
		location_collection = get_collection(db_prod, "locations_data")

		# Save all new locations with associated articles
		for n, locations in enumerate(df["locations"]):
			for i, location in enumerate(locations):

				location_collection.update_one(
					{'value': location},
					{'$addToSet': {'articles': df["content_id"][n]},
	  				 '$setOnInsert': {'neighborhood': df["neighborhoods"][n][i]},
					 '$setOnInsert': {'tract': df["tracts"][n][i]},
					 '$setOnInsert': {'coordinates': df["coordinates"][n][i]}
	  				},
					upsert = True 
				)          
		print("[INFO] Locations Successfully inserted!")
	except Exception as err:
		raise Exception(f"[Error!] Error in sending Locations Data\nError: {err}")
	return

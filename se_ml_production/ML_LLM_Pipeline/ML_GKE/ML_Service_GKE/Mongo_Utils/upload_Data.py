import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import secret
from pymongo.errors import BulkWriteError

from Mongo_Utils.mongo_neighborhoods import neigh_tract_dict
from Mongo_Utils.mongo_funcs import get_collection, convert_to_datesum
from Mongo_Utils.mongo_tracts import update_tracts, create_gen_tract
from Model_Utils.location_Utils import best_location

# ==== Packing Funcs ====
def send_to_production(client, df, org_key):
	try:
		db_prod = client[secret.db_name]

		# Pack and send all articles
		pack_articles(db_prod, df, org_key)
		pack_neighborhoods(db_prod, df, org_key)
		pack_topics(db_prod, df, org_key)
		pack_tracts(db_prod, df, org_key)
		pack_locations(db_prod, df, org_key)

	except Exception as err:
		print(f"[Error!] Error in sending data to MongoDB Prod DB\nError: {err}")
		raise Exception("Fatal Error in sending to production")
	return

# Pack the article data and send to MongoDB
def pack_articles(db_prod, df, org_key):
	try:
		df['dateSum'] = df['pub_date'].apply(convert_to_datesum)

		article_payload = df.to_dict(orient='records')
		collection_name = "articles_data"
		articles_collection = get_collection(db_prod, f"{collection_name}_{org_key}")

		try: 
			articles_collection.insert_many(article_payload, ordered=False)
		except BulkWriteError as bwe:
			# Handle duplicate key errors
			write_errors = bwe.details.get('writeErrors', [])
			duplicates = [error['op'] for error in write_errors if error['code'] == 11000]
			if duplicates:
				print(f"[WARNING] Skipped {len(duplicates)} duplicate articles.")
			else:
				raise bwe
		return
	except Exception as err:
		raise Exception(f"[Error!] Error in sending Article Data\nError: {err}")
	return

# Pack the neighborhood data and send to MongoDB
def pack_neighborhoods(db_prod, df, org_key):
	try:
		collection_name = "neighborhood_data"
		neigh_collection = get_collection(db_prod, f"{collection_name}_{org_key}")
		
		
		# Save all new neighborhoods with associated tracts and articles
		for n, neighborhoods in enumerate(df['neighborhoods']):
			for i, neighborhood in enumerate(neighborhoods):				
				# More convoluted than it should be. There's a bug with addToSet so this is a workaround
				current_doc = neigh_collection.find_one({'value': neighborhood})
				update_data = {}
				
				if current_doc:
					if df["_id"][n] not in current_doc.get('articles', []):
						update_data['articles'] = df["_id"][n]
					if df['tracts'][n][i] not in current_doc.get('tracts', []):
						update_data['tracts'] = df['tracts'][n][i]
					if df['locations'][n][i] not in current_doc.get('locations', []):
						update_data['locations'] = df['locations'][n][i]
				else:
					# Initialize values
					update_data['articles'] = df["_id"][n]
					update_data['tracts'] = df['tracts'][n][i]
					update_data['locations'] = df['locations'][n][i]

				# Update the document if there's anything to update
				if update_data:
					neigh_collection.update_one(
						{'value': neighborhood},
						{
							'$push': update_data
						}, upsert=True
					)
	except Exception as err:
		raise Exception(f"[ERROR]  Error in sending Neighborhood Data\nError: {err}")
	return

# Pack the topics data and send to MongoDB
def pack_topics(db_prod, df, org_key):
	try:
		collection_name = "topics_data"
		topic_collection = get_collection(db_prod, f"{collection_name}_{org_key}")

		# Save all new topics with associated articles
		for n, topic in enumerate(df["openai_labels"]):
			topic_collection.update_one(
				{'value': topic},
				{'$addToSet': {'articles': df["_id"][n]}},
				upsert = True 
			)          
	except Exception as err:
		raise Exception(f"[ERROR]  Error in sending Topics Data\nError: {err}")
	return

# Pack the tracts data and send to MongoDB
def pack_tracts(db_prod, df, org_key):
	try:
		collection_name = "tracts_data"
		tract_collection = get_collection(db_prod, f"{collection_name}_{org_key}")

		# Save all new tracts with associated articles and neighborhoods
		for n, tracts in enumerate(df['tracts']):
			for i, tract in enumerate(tracts):
				if tract_collection.find_one({'tract': tract}):
					tract_collection.update_one(
					{'tract': tract},
					{
					 '$push': {'articles': df['_id'][n]},
	  				 '$addToSet': {'locations': df['locations'][n][i]},
					}, upsert=True
    				) 
				else:
					update_tracts(tract_collection, tract, df["neighborhoods"][n][i], df["counties"][n][i], df["states"][n][i], df["cities"][n][i], df['_id'][n], df['locations'][n][i]) 
				
				# Create and update global tract for neighborhood/city
				gen_tract = df["neighborhoods"][n][i]
				if tract_collection.find_one({'tract': gen_tract}):
					tract_collection.update_one(
					{'tract': gen_tract},
					{
					 '$push': {'articles': df['_id'][n]},
	  				 '$addToSet': {'locations': df['locations'][n][i]},
					}, upsert=True
					)
				else:
					create_gen_tract(tract_collection, gen_tract, gen_tract, df["counties"][n][i], df["states"][n][i], df["cities"][n][i], df['_id'][n], df['locations'][n][i])

	except Exception as err:
		raise Exception(f"[ERROR] Error in sending Tracts Data\nError: {err}")
	return


# Pack the locations data and send to MongoDB
def pack_locations(db_prod, df, org_key):
	try:
		collection_name = "locations_data"
		location_collection = get_collection(db_prod, f"{collection_name}_{org_key}")

		# Save all new locations with associated articles
		for n, locations in enumerate(df["locations"]):
			for i, location in enumerate(locations):
				coordinates = df["coordinates"][n][i]

				# Try to find the location document based on coordinates
				location_doc = location_collection.find_one({'coordinates': coordinates})
				if location_doc:
					all_locations = location_doc.get('all_locations', [])
					if location not in all_locations:
						all_locations.append(location)
						best_loc = best_location(all_locations)
						current_loc = location_doc.get('value', None)
						
						# Update location document with the new best location if needed
						update_fields = {
							'$addToSet': {
								'articles': df["_id"][n],
								'all_locations': location
							}
						}
						if best_loc != current_loc:
							update_fields['$set'] = {'value': best_loc}

						location_collection.update_one(
							{'coordinates': coordinates},
							update_fields
						)
					else:
						# Only add the article if the location already exists
						location_collection.update_one(
							{'coordinates': coordinates},
							{'$addToSet': {'articles': df["_id"][n]}}
						)
				else:
					# Insert a new document with the specified fields
					location_collection.insert_one({
						'coordinates': coordinates,
						'articles': [df["_id"][n]],
						'all_locations': [location],
						'value': location,
						'neighborhood': df["neighborhoods"][n][i],
						'tract': df["tracts"][n][i],
						'city': df["cities"][n][i],
						'state': df["states"][n][i]
					})

          
	except Exception as err:
		raise Exception(f"[ERROR]  Error in sending Locations Data\nError: {err}")
	return


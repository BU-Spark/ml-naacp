import secret
from datetime import datetime
from global_state import global_instance
from Mongo_Utils.mongo_funcs import connect_MongoDB_Prod

def convert_to_datesum(s):
	date_formatted = s.replace('-', '').replace(' ', '').replace(':', '')

	year = date_formatted[-4:]
	month_num = date_formatted[3:6]
	month = str(datetime.strptime(month_num, "%b").month)
	day = date_formatted[6:8]

	if (int(month) <= 9):
		year = str(year) + "0"
		return int(year + month + day)

	return int(year + month + day)

def string_to_list(s):
    if(s != ''):
        return [s]
    else:
        return []  # Return the string as a single-element list

def addExistingTracts(tract_collection):
	tracts_list = []

	for tract_key in global_instance.get_data("census_base").old_census_tracts['tracts_filter'].keys():
		tracts_list.append(global_instance.get_data("census_base").old_census_tracts['tracts_filter'][tract_key])

	tract_collection.insert_many(tracts_list)

def send_Discarded(client, discard_list):
	try:
		# Pack and send all articles
		db_prod = client[secret.db_name]
		
		discarded_collection_name = "discarded"
		discarded_collection = db_prod[discarded_collection_name]

		for discarded_article in discard_list:
			if (discarded_collection.find_one({'uploadID': global_instance.get_data("upload_id")})):
				discarded_collection.find_one_and_update(
					{'uploadID': global_instance.get_data("upload_id")},
					{'$addToSet': {'content_ids': discarded_article}},
				)
			else:
				new_discard = {
					'userID': global_instance.get_data("userID"),
					'uploadID': global_instance.get_data("upload_id"),
					'content_ids': [discarded_article]
				}
				discarded_collection.insert_one(new_discard)
		print("[INFO] Discarded List Successfully inserted!")
		return
	except Exception as err:
		print(f"[Error!] Error in sending data to MongoDB Prod DB\nError: {err}")
		raise Exception("Fatal Error in sending to production")
	return

# ==== Packing Funcs ====
def send_to_production(client, df):
	try:
		db_prod = client[secret.db_name]

		# Pack and send all articles
		pack_articles(db_prod, df)
		pack_neighborhoods(db_prod, df)
		pack_topics(db_prod, df)
		pack_tracts(db_prod, df)

	except Exception as err:
		print(f"[Error!] Error in sending data to MongoDB Prod DB\nError: {err}")
		raise Exception("Fatal Error in sending to production")
	return

def pack_articles(db_prod, df):
	try:
		article_payload = []
		articles_collection_name = "articles_data"
		articles_collection = db_prod[articles_collection_name]

		collection_list = db_prod.list_collection_names()

		if articles_collection_name not in collection_list:
		    db_prod.create_collection(articles_collection_name)
		    print(f"[INFO] Collection '{articles_collection_name}' created.")

		article_df = df.set_index('id')
		article_dict = article_df.T.to_dict('dict')

		for article_key in article_dict.keys():
		    article = article_dict[article_key]
		    if ('openai_labels' not in article):
		        article["openai_labels"] = []
		    else:
		        article["openai_labels"] = string_to_list(article["openai_labels"])
		    article["dateSum"] = convert_to_datesum(article["pub_date"])
		    article_payload.append(article)

		articles_collection.insert_many(article_payload)
		print("[INFO] Articles Successfully inserted!")
		return
	except Exception as err:
		raise Exception(f"[Error!] Error in sending Article Data\nError: {err}")
	return

# Quick fix, but it will work | Based off of Boston Census 2020 data
neigh_tract_dict = {
	"Fenway" : ["010103", "010104", "010204", "010408", "010404", "010403", "981501", "010405", "010206", "010205"],
	"Downtown": ["030302", "070202", "070102", "030301", "070104", "070103", "070201"],
	"Beacon Hill": ["020200", "020302", "020101", "981700"],
	"Dorchester" : [
	"092400", "091400", "090300", "091800", "092300", "100601", "090901", 
	"100400", "090100", "091001","090200", "100200", "091700", "092200", "090700",
	"091500", "091300", "100300", "100100", "092000", "100500", "100800", "100603",
	"091200", "100700", "092101", "091900", "091600", "091100"
	],
	"Mattapan": ["100900", "101002", "101102", "981100", "101001","101101"],
	"Jamaica Plain": [
	"120103", "981800", "110105", "120600", "120700", "120301", "081200", "120105","081101",
	"981000", "120500", "120104", "120201", "110106", "081301", "120400"
	],
	"Roslindale": ["110502", "110104", "110501", "110401", "140106", "110301", "110607", "110403","110201"],
	"Roxbury": [
	"081500", "080500", "070801", "080100", "081800", "980300", "082000", "080601", "081700", "080300",
	"090600", "081400", "090400", "070901", "082100", "081900", "081302","080401"
	],
	"West End": ["020304", "020301", "020305"],
	"Longwood": ["010300", "081001"],
	"South Boston": ["061101", "060700", "060101", "061201", "061000", "060800", "981201", "060200", "061202", "060400", "061203", "060301", "060601", "060501"],
	"Back Bay": ["010702", "010701", "010802", "010801", "010500", "010600"],
	"Charlestown": ["040100", "040300", "040401", "040600", "040801", "040200"],
	"Allston": ["000604", "000804", "000703", "000704", "000806", "000101", "000807", "000701", "000805"],
	"Hyde Park": ["140107", "140201", "140105", "980700", "140300", "140202", "140400", "140102"],
	"East Boston": ["050500", "050600", "981502", "050101", "981300", "050901", "050300", "050700", "050400", "051000", "981600", "051200", "050200", "051101"],
	"South End": ["070301", "070302", "070502", "070501", "071101", "070600", "070700", "070902", "070802", "071201", "070402"],
	"West Roxbury": ["980900", "130406", "981900", "130404", "110601", "130300", "130402", "130200", "130101"],
	"South Boston Waterfront": ["981202", "060602", "060603", "061204", "060604"],
	"North End": ["030200", "030100", "030500", "030400"]
}

def pack_neighborhoods(db_prod, df):
	try:
		neigh_collection_name = "neighborhood_data"
		neigh_collection = db_prod[neigh_collection_name]

		collection_list = db_prod.list_collection_names()

		if neigh_collection_name not in collection_list:
			db_prod.create_collection(neigh_collection_name)
			print(f"[INFO] Collection '{neigh_collection_name}' created.")

		neighborhood_list = df['neighborhoods'].to_numpy()
		tagging_list = df['content_id'].to_numpy()

		for n_idx in range(len(neighborhood_list)):
			neigh_list = neighborhood_list[n_idx]

			for neigh in neigh_list:
				# Here we update the tags/articles by neighborhood
				if (neigh not in neigh_tract_dict.keys()):
					neigh_collection.find_one_and_update(
						{'value': neigh},
						{
							'$addToSet': {'articles': tagging_list[n_idx]},
							'$setOnInsert': {'tracts': []}
						},
						upsert = True # Creates a new document of it if it doesn't exist
					)
					continue

				neigh_collection.find_one_and_update(
					{'value': neigh},
					{
						'$addToSet': {'articles': tagging_list[n_idx]},
						'$setOnInsert': {'tracts': neigh_tract_dict[neigh]}
					},
					upsert = True # Creates a new document of it if it doesn't exist
				)       
		print("[INFO] Neighborhoods Successfully inserted!")
	except Exception as err:
		raise Exception("[Error!] Error in sending Neighborhood Data\nError: {err}")
	return

def pack_topics(db_prod, df):
	try:
		topics_collection_name = "topics_data"
		topic_collection = db_prod[topics_collection_name]

		collection_list = db_prod.list_collection_names()

		if topics_collection_name not in collection_list:
			db_prod.create_collection(topics_collection_name)
			print(f"[INFO] Collection '{topics_collection_name}' created.")

		topics_list = df['position_section'].to_numpy()
		tagging_list = df['content_id'].to_numpy()

		for n_idx in range(len(topics_list)):
			topic = topics_list[n_idx]
		    
			# Here we update the tags/articles by Topics
			topic_collection.find_one_and_update(
				{'value': topic},
				{'$addToSet': {'articles': tagging_list[n_idx]}},
				upsert = True # Creates a new document of it if it doesn't exist
			)          
		print("[INFO] Topics Successfully inserted!")
	except Exception as err:
		raise Exception("[Error!] Error in sending Topics Data\nError: {err}")
	return

def pack_tracts(db_prod, df):
	try:
		tract_collection_name = "tracts_data"
		tract_collection = db_prod[tract_collection_name]

		collection_list = db_prod.list_collection_names()

		# Check for existence of collection
		collection_list = db_prod.list_collection_names()

		if tract_collection_name not in collection_list:
			db_prod.create_collection(tract_collection_name)
			print(f"[INFO] Collection '{tract_collection_name}' created.")
			addExistingTracts(tract_collection) # We add existing tracts from the old census tracts

		tracts_lists = df['tracts'].to_numpy()
		tagging_list = df['content_id'].to_numpy()

		for n_idx in range(len(tracts_lists)):
			tract_list = tracts_lists[n_idx]

			for tract in tract_list:
				# Here we update the tags/articles by tracts
				if tract_collection.find_one({'tract': tract}):
					tract_collection.find_one_and_update(
						{'tract': tract},
						{'$addToSet': {'articles': tagging_list[n_idx]}},
					)
				else: # We didn't find one and we have to label it as unknown
					unknown_tract = {
						'tract': tract,
						'neighborhood': "unknown",
						'articles': [tagging_list[n_idx]]
					}
					tract_collection.insert_one(unknown_tract)     
		print("[INFO] Tracts Successfully inserted!")
	except Exception as err:
		raise Exception("[Error!] Error in sending Tracts Data\nError: {err}")
	return


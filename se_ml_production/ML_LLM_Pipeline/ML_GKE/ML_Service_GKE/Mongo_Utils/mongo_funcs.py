import secret
from pymongo import MongoClient
from datetime import datetime

def update_job_status(client, upload_id, user_id, timestamp, article_cnt, status, message):
    try:
        db = client[secret.db_name]
        
        upload_collection = get_collection(db, "uploads")

        updated_fields = {
            'status': status,
            'message': message,
            'article_cnt': article_cnt
        }

        upload_collection.update_one(
            {'uploadID': upload_id},
            {'$set': updated_fields, '$setOnInsert': {'userID': user_id, 'timestamp': str(timestamp)}},
            upsert=True
        )

        print(f"[INFO] Job: {upload_id} is now of status {status}.")
    except Exception as err:
        print(f"[Job Error!] {err}")
        raise Exception("[Job Error!] Failed to update Job Status")
    return 

# Get the collection from the database
def get_collection(db_prod, collection_name):
	collection_list = db_prod.list_collection_names()

	# Initialize the collection if it doesn't exist
	if collection_name not in collection_list:
		db_prod.create_collection(collection_name)
		print(f"[INFO] Collection '{collection_name}' created.")

	return db_prod[collection_name]

# Convert date to a number for easier comparison
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
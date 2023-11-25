import secret
from pymongo import MongoClient

def connect_MongoDB_Prod():
    try:
        client = MongoClient(secret.MONGO_URI_NAACP)
        db = client['se_naacp_db']
        return db
    except Exception as err:
        raise Exception("[Fatal Error!] Failed to Connect to MongoDB Production Database. [No retry implemented]")
    return 

def update_job_status(upload_id, user_id, timestamp, article_cnt, status, message):
    try:
        client = MongoClient(secret.MONGO_URI_NAACP)
        db = client['se_naacp_db']
        
        upload_collection = db["uploads"]
        if (upload_collection.find_one({'uploadID': upload_id})):
            upload_collection.find_one_and_update(
                {'uploadID': upload_id},
                {'$set': {'status': status}}
            )
            upload_collection.find_one_and_update(
                {'uploadID': upload_id},
                {'$set': {'message': message}}
            )
            upload_collection.find_one_and_update(
                {'uploadID': upload_id},
                {'$set': {'article_cnt': article_cnt}}
            )
        else: # We didn't find one and we have to label it as unknown
            new_status = {
                "uploadID": upload_id, 
                "userID":user_id,
                "timestamp": str(timestamp), 
                "article_cnt": article_cnt, 
                "status" : status,
                "message": message
            }
            upload_collection.insert_one(new_status)
        print(f"[INFO] Job: {upload_id} is now of status {status}.")
    except Exception as err:
        print(f"[Job Error!] {err}")
        raise Exception("[Job Error!] Failed to update Job Status")
    return 
import secret
from pymongo import MongoClient


def update_job_status(client, upload_id, user_id, org_id, filename, timestamp, article_cnt, status, message):
    try:
        db = client[secret.db_name]
        
        upload_collection = db["uploads"]

        updated_fields = {
            'status': status,
            'message': message,
            'article_cnt': article_cnt
        }

        upload_collection.update_one(
            {'uploadID': upload_id},
            {'$set': updated_fields, '$setOnInsert': {
                'userID': user_id, 
                'orgID': org_id,
                'filename': filename,
                'timestamp': str(timestamp)}},
            upsert=True
        )

        print(f"[INFO] Job: {upload_id} is now of status {status}.")
    except Exception as err:
        print(f"[Job Error!] {err}")
        raise Exception("[Job Error!] Failed to update Job Status")
    return 
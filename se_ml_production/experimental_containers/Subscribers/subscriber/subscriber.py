"""
Here we have a subscriber listen for any messages and acknowledge. We also simulating doing work.
"""
import os
import time
import json
import threading
import pandas as pd
from io import BytesIO
from queue import Queue
from google.cloud import storage
from google.cloud import pubsub_v1


# Use a thread-safe queue instead of a list
message_queue = Queue()

def process_data():
    def get_csv(db, target_csv_file): # Make this a private function
        """
        Loads the csv from the bucket
        """
        try:
            if (db == None):
                raise Exception("No database was given!")
            
            bucket = db.bucket("shiply_csv_bucket")
            blob = bucket.blob(f"data/{target_csv_file}")
            data = blob.download_as_bytes()
            df = pd.read_csv(BytesIO(data))
                
            return df
        except Exception as err:
            print(f"Error fetching csv data!\nError: {err}")
            raise Exception("Fatal Error in fetching csv data!")
        return

    while True:
        # Wait until a message is available
        message_data = message_queue.get()

        print(f"[INFO] Processing Data: {message_data}\n")
        message_data = json.loads(message_data)
        target_csv_file = message_data["upload_id"] + "-" + message_data["userID"] + ".csv"

        client = storage.Client()
        df = get_csv(client, target_csv_file)

        print(df)

        print("Doing some processing...")
        time.sleep(10)  # Sleep 5 seconds to simulate processing
        print("Processing Finished.")

        # Signal that the processing is complete
        message_queue.task_done()

def callback(message):
    message_queue.put(message.data.decode('utf-8'))
    message.ack()  # Acknowledge the message
    print(f"[INFO] Acknowledged Message Content: {message.data.decode('utf-8')}\n")
    return

def main():
    subscription_id = "shiply_upload_csv-sub"
    project_id = "special-michelle"

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_id)
    
    # Start a separate thread for processing data
    processing_thread = threading.Thread(target=process_data)
    processing_thread.daemon = True  # Daemonize thread
    processing_thread.start()

    # Subscribe and listen for messages
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    print("Initialized Application. Listening for messages...")

    try:
        # This will block until an exception is thrown (like KeyboardInterrupt)
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()
        print("Subscription canceled.")

    print("Exiting application.")

main()

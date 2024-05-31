import os
import json
import threading
import pandas as pd
from io import BytesIO
from queue import Queue
from google.cloud import storage
from google.cloud import pubsub_v1

import nltk
from ML_Entry import run_pipeline
from global_state import global_instance
from Mongo_Utils.mongo_funcs import connect_MongoDB_Prod
from bootstrappers import bootstrap_pipeline, validate_bootstrap, bootstrap_MongoDB_Prod

# Use a thread-safe queue instead of a list
message_queue = Queue()

def startup_event():
    """
    We store all global variables needed by all functions through our global state file
    """
    try:
        # Main pipeline Boostrap
        (year,
        dsource,
        dname,
        state,
        drop_geos,
        mappings,
        census_base,
        heir_data,
        saved_geocodes,
        nlp_ner,
        nlp_topic,
        db,
        db_manager) = bootstrap_pipeline()

        global_instance.update_data("year", year)
        global_instance.update_data("dsource", dsource)
        global_instance.update_data("dname", dname)
        global_instance.update_data("state", state)
        global_instance.update_data("drop_geos", drop_geos)
        global_instance.update_data("mappings", mappings)
        global_instance.update_data("census_base", census_base)
        global_instance.update_data("heir_data", heir_data)
        global_instance.update_data("saved_geocodes", saved_geocodes)
        global_instance.update_data("nlp_ner", nlp_ner)
        global_instance.update_data("nlp_topic", nlp_topic)
        global_instance.update_data("db_manager", db_manager)

        validate_bootstrap(
            year,
            dsource,
            dname,
            state,
            drop_geos,
            mappings,
            census_base,
            heir_data,
            saved_geocodes,
            nlp_ner,
            nlp_topic,
            db,
            db_manager
        )

        nltk.download('punkt')

        print("Deployment Test Complete with no errors")

        # MongoDB Bootstrap
        defined_collection_names = ["uploads", "discarded"]
        db_prod = connect_MongoDB_Prod()
        db_manager = global_instance.get_data("db_manager")
        # We then create our first MongoDB connection
        db_manager.init_connection(uri=os.environ['MONGO_URI_NAACP'])

        db_manager.run_job(
            bootstrap_MongoDB_Prod, 
            db_manager.act_con[0]['connection'], # Argument 1 (1st connection)
            defined_collection_names, # Argument 2
            connection_obj=db_manager.act_con[0]
            )
    except Exception as e:
        print(f"[Error!] FATAL ERROR! | {e}")
        raise

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
        try:
            # Wait until a message is available
            message_data = message_queue.get()

            print(f"\n[INFO] Processing Data: {message_data}\n")
            message_data = json.loads(message_data)
            target_csv_file = message_data["upload_id"] + "-" + message_data["userID"] + ".csv"

            client = storage.Client()
            df = get_csv(client, target_csv_file)

            print("[DEBUG] Dataframe Content:\n")
            print(df)

            # We run the ML Pipeline here
            run_pipeline(df, message_data["upload_id"], message_data["userID"], message_data["uploadTimeStamp"])

            # Signal that the processing is complete
            message_queue.task_done()
        except Exception as e:
            print(f"[Error!] Fatal error in processing data! | {e}")
            raise

def callback(message):
    message_queue.put(message.data.decode('utf-8'))
    message.ack()  # Acknowledge the message
    print(f"\n[INFO] Acknowledged Message Content: {message.data.decode('utf-8')}\n")
    return

def main():
    try:
        startup_event() # Bootstrap the entire container

        # Then we start the subscription
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
        print("[INFO] Finished creating subscription. Container is now listening...\n\n")
        streaming_pull_future.result() # This thread is blocked until a exception is thrown
    except Exception as e:
        streaming_pull_future.cancel()
        print("[INFO] Subscription canceled!")
        print(f"[Error!] FATAL ERROR! | {e}")
        raise
    return

    


main()




















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

    return

def main():
    try:
        startup_event() # Bootstrap the entire container
    except Exception as e:
        print(f"[Error!] FATAL ERROR! | {e}")
        raise
    return

    

main()




















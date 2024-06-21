import json
import pandas as pd
from bson import ObjectId

class neighborhood_mapping():
    def __init__(self, db, fetch_arr):
        self.db = db
        self.fetch_arr = fetch_arr
        self.load_mappings()
    
    def load_mappings(self):
        try:
            if (self.db == None):
                raise Exception("No database was given!")
                
            if (self.fetch_arr[0]):
                bucket = self.db.bucket("ml_naacp_model_data")
                blob = bucket.blob("Entity_Recognition_Pipeline_Data/tracts-neighbors.json")
                blob.download_to_filename("./geodata_prod/tracts-neighbors.json")

            if (self.fetch_arr[1]):
                bucket = self.db.bucket("ml_naacp_model_data")
                blob = bucket.blob("Entity_Recognition_Pipeline_Data/blocks-neighbors.json")
                blob.download_to_filename("./geodata_prod/blocks-neighbors.json")

            block_map_db = json.load(open("./geodata_prod/blocks-neighbors.json"))
            tract_map_db = json.load(open("./geodata_prod/tracts-neighbors.json"))
            
            self.tract_mapping = tract_map_db
            self.block_mapping = block_map_db
        except Exception as err:
            print(f"Error loading neighborhood mapping class!\nError: {err}")
            raise Exception("Fatal Error in Class Construction!")
        return

    def tract_to_neighborhood(self, tract):
        # given a census tract return the boston neighborhood it is in 
        return self.tract_mapping[tract]

    def block_to_neighborhood(self, block):
        # given a census block return the boston neighborhood it is in 
        return self.block_mapping(block)
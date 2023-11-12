import pandas as pd
from bson import ObjectId

class neighborhood_mapping():
    def __init__(self, db):
        self.db = db
        self.load_mappings()
    
    def load_mappings(self):
        try:
            if (self.db == None):
                raise Exception("No database was given!")

            # load census tract to boston neighborhood mapping 
            # load census block to boston neighborhood mapping 
            tract_map_db = self.db['Entity_Recognition_Pipeline_Data'].find_one(
                {"_id": ObjectId("654ad02fd4c2ceb2f5d5ddfc")}
            )

            block_map_db = self.db['Entity_Recognition_Pipeline_Data'].find_one(
                {"_id": ObjectId("654ad118d4c2ceb2f5d5ddfd")}
            )
            
            if (tract_map_db == None or block_map_db == None):
                raise Exception(f"tract_map_db or block_map_db yielded None respectively. Length: [{len(tract_map_db)},{(block_map_db)}]")

            del tract_map_db['_id']
            del block_map_db['_id']
            
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
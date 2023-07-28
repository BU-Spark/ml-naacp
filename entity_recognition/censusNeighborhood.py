import pandas as pd
import json 

class neighborhood_mapping():
    def __init__(self):
        self.load_mappings()
    
    def load_mappings(self):
        # load census tract to boston neighborhood mapping 
        # load census block to boston neighborhood mapping 
        self.tract_mapping = json.load(open("./entity_recognition/geo-data/tracts-neighbors.json"))
        self.block_mapping = json.load(open("./entity_recognition/geo-data/blocks-neighbors.json"))

    def tract_to_neighborhood(self, tract):
        # given a census tract return the boston neighborhood it is in 
        return self.tract_mapping[tract]

    def block_to_neighborhood(self, block):
        # given a census block return the boston neighborhood it is in 
        return self.block_mapping(block)
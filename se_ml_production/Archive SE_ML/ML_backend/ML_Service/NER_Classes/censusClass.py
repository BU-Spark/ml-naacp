"""
This is a class that takes in two csv's. One csv for census data and one for census data overall by neighborhoods.
As of this creation of this class, we are using the Boston Census 2020 data. Please make sure that future 
versions follow the same path.
"""
import json
import pandas as pd
from bson import ObjectId

class census_data():
    def __init__(self, db, fetch_arr):
        self.db = db
        self.fetch_arr = fetch_arr
        self.load_census_neigh_data()
        self.load_old_census_data()
        self.load_census_data()
        
    def load_old_census_data(self):
        try:
            if (self.db == None):
                raise Exception("No database was given!")
                
            if (self.fetch_arr[2]):
                bucket = self.db.bucket("ml_naacp_model_data")
                blob = bucket.blob("Topic_Modeling_Pipeline_Data/census.json")
                blob.download_to_filename("./geodata_prod/census.json")

            old_census_tracts_db = json.load(open("./geodata_prod/census.json"))

            self.old_census_tracts = old_census_tracts_db
        except Exception as err:
            print(f"Error loading Census Data class!\nError: {err}")
            raise Exception("Fatal Error in Class Construction!")
        return
    
    def load_census_data(self, fetch=True):
        try:
            if (self.db == None):
                raise Exception("No database was given!")

            if (self.fetch_arr[1]):
                bucket = self.db.bucket("ml_naacp_model_data")
                blob = bucket.blob("Topic_Modeling_Pipeline_Data/census_2020.csv")
                blob.download_to_filename("./geodata_prod/census_2020.csv")

            census_tracts_df = pd.read_csv("./geodata_prod/census_2020.csv")
            
            self.census_tracts = self.process_census_data(census_tracts_df)
        except Exception as err:
            print(f"Error loading Census Data class!\nError: {err}")
            raise Exception("Fatal Error in Class Construction!")
        return
        
    def load_census_neigh_data(self, fetch=True):
        try:
            if (self.db == None):
                raise Exception("No database was given!")
                
            if (self.fetch_arr[0]):
                bucket = self.db.bucket("ml_naacp_model_data")
                blob = bucket.blob("Topic_Modeling_Pipeline_Data/census_2020_neigh.csv")
                blob.download_to_filename("./geodata_prod/census_2020_neigh.csv")

            census_neigh_data_df = pd.read_csv("./geodata_prod/census_2020_neigh.csv")

            self.census_neighbourhoods = self.process_census_neigh_data(census_neigh_data_df)
        except Exception as err:
            print(f"Error loading Census Data class!\nError: {err}")
            raise Exception("Fatal Error in Class Construction!")
        return

    def process_census_data(self, df):
        demographics = df.iloc[:,11:20]
        geoid_tract = df['GEOCODE']
        tract = df['TRACT']

        concat_pd = pd.concat([tract, geoid_tract, demographics], axis=1)
        concat_pd.drop(concat_pd.index[0], inplace=True)
        concat_pd = concat_pd.rename(
            columns={
                'TRACT': 'tract', 
                'GEOCODE': 'geoid_tract',
                'P0020001': 'total'        
            }
        )
        return concat_pd

    # A function for the future
    def process_census_neigh_data(self, df):
        df = df.iloc[:,:7]
        df = df.rename(
            columns={
                'tract20_nbhd': 'Neighborhood',       
            }
        )
        df.drop(df.index[0], inplace=True)
        return df
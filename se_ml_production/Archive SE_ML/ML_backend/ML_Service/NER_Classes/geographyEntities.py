import json
import pandas as pd
from bson import ObjectId

class geography():
    def __init__(self, db, fetch_arr):
        self.db = db
        self.fetch_arr = fetch_arr
        self.load_mass_town_entities()
        self.load_state_entities()
        self.load_saved_geocodes()
        self.load_org_entities()

    def load_saved_geocodes(self):
        try:
            if (self.db == None):
                raise Exception("No database was given!")
                
            if (self.fetch_arr[0]):
                bucket = self.db.bucket("ml_naacp_model_data")
                blob = bucket.blob("Entity_Recognition_Pipeline_Data/saved-geocodes.json")
                blob.download_to_filename("./geodata_prod/saved-geocodes.json")
                
            saved_geocodes_db = json.load(open("./geodata_prod/saved-geocodes.json"))
            
            self.saved_geocodes = saved_geocodes_db
        except Exception as err:
            print(f"Error loading geography class!\nError: {err}")
            raise Exception("Fatal Error in Class Construction!")
        return
        
    def load_state_entities(self, fetch=True):
        try:
            if (self.db == None):
                raise Exception("No database was given!")

            if (self.fetch_arr[1]):
                bucket = self.db.bucket("ml_naacp_model_data")
                blob = bucket.blob("Entity_Recognition_Pipeline_Data/states.csv")
                blob.download_to_filename("./geodata_prod/states.csv")
            
            states = pd.read_csv("./geodata_prod/states.csv")
            
            states_set = set()
            for idx in range(len(states)):
                tup = (states['state'][idx], 'LOC')
                states_set.add(tup)
            self.state_entities = states_set
            
        except Exception as err:
            print(f"Error loading geography class!\nError: {err}")
            raise Exception("Fatal Error in Class Construction!")
        return
    
    def load_mass_town_entities(self, fetch=True):
        try:
            if (self.db == None):
                raise Exception("No database was given!")
                
            if (self.fetch_arr[2]):
                bucket = self.db.bucket("ml_naacp_model_data")
                blob = bucket.blob("Entity_Recognition_Pipeline_Data/mass-towns.csv")
                blob.download_to_filename("./geodata_prod/mass-towns.csv")
                
            towns = pd.read_csv("./geodata_prod/mass-towns.csv")

            towns_set = set()
            for idx in range(len(towns)):
                tup = (towns['town'][idx], 'LOC')
                towns_set.add(tup)
            self.mass_town_entities = towns_set
        except Exception as err:
            print(f"Error loading geography class!\nError: {err}")
            raise Exception("Fatal Error in Class Construction!")
        return
    
    def load_org_entities(self):
        self.org_entities = (('GBH News', 'ORG'), ('Boston Public Radio', 'ORG'), 
                             ('Supreme Court', 'ORG'), ('New York Times', 'ORG'), 
                             ('Washington Post', 'ORG'), ('CNN', 'ORG'), 
                             ('NPR', 'ORG'), ('Associated', 'ORG'), 
                             ('Press', 'ORG'), ('Senate', 'ORG'), 
                             ('Associated Press', 'ORG'), ('AP', 'ORG'), 
                             ('ABC News', 'ORG'),('CSS', 'ORG'), 
                             ('Philadelphia Inquirer', 'ORG'), ('House', 'ORG'),
                             ('Congress', 'ORG'), ('Worcester', 'ORG'),
                             ('FBI', 'ORG'), ('Homeland Security Department', 'ORG'),
                             ('CDC', 'ORG'),('Fox News', 'ORG'),('The Washington Post', 'ORG'),
                             ('States', 'LOC'), ('S.', 'LOC'), ('Massachusetts', 'ORG'),
                             ('White House', 'ORG'), ('High School', 'ORG'),
                             ('MIT', 'ORG'), ('Harvard University', 'ORG'),
                             ('White House', 'LOC'),('Greater Boston', 'LOC'),
                             ('New England', 'LOC'))
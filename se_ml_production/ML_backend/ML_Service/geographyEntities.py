import pandas as pd
from bson import ObjectId

class geography():
    def __init__(self, db):
        self.db = db
        self.load_geographies()
        self.load_org_entities()
        self.load_saved_geocodes()

    def load_saved_geocodes(self):
        try:
            if (self.db == None):
                raise Exception("No database was given!")
                
            saved_geocodes_db = self.db['Entity_Recognition_Pipeline_Data'].find_one(
                {"_id": ObjectId("654ad709d4c2ceb2f5d5de00")}
            )
            
            if (saved_geocodes_db == None):
                raise Exception(f"saved_geocodes_db yielded None. Length: [{len(saved_geocodes_db)}]")

            del saved_geocodes_db['_id']
            
            self.saved_geocodes = saved_geocodes_db
        except Exception as err:
            print(f"Error loading geography class!\nError: {err}")
            raise Exception("Fatal Error in Class Construction!")
        return

    def load_geographies(self):
        # static data, states, towns, orgs in entity output format to filter out of geocoding results 
        self.load_state_entities()
        self.load_mass_town_entities()
    
    def load_state_entities(self):
        try:
            if (self.db == None):
                raise Exception("No database was given!")
                
            state_entities_db = self.db['Entity_Recognition_Pipeline_Data'].find_one(
                {"_id": ObjectId("654ae0edd4c2ceb2f5d5de33")}
            )
            
            if (state_entities_db == None):
                raise Exception(f"state_entities_db yielded None. Length: [{len(state_entities_db)}]")

            csv_data = state_entities_db["csv_data"]
            state_entities_df = pd.DataFrame(csv_data)

            states_set = set()
            for idx, row in state_entities_df.iterrows():
                tup = (row.iloc[0], 'LOC')
                states_set.add(tup)              
            self.state_entities = states_set
        except Exception as err:
            print(f"Error loading geography class!\nError: {err}")
            raise Exception("Fatal Error in Class Construction!")
        return
    
    def load_mass_town_entities(self):
        try:
            if (self.db == None):
                raise Exception("No database was given!")
                
            mass_town_entities_db = self.db['Entity_Recognition_Pipeline_Data'].find_one(
                {"_id": ObjectId("654ae26ed4c2ceb2f5d5de34")}
            )
            
            if (mass_town_entities_db == None):
                raise Exception(f"mass_town_entities_db yielded None. Length: [{len(mass_town_entities_db)}]")

            csv_data = mass_town_entities_db["csv_data"]
            mass_town_entities_df = pd.DataFrame(csv_data)

            towns_set = set()
            for idx, row in mass_town_entities_df.iterrows():
                tup = (row.iloc[0], 'LOC')
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
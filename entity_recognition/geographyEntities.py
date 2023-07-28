import pandas as pd
import json

class geography():
    def __init__(self):
        self.load_geographies()
        self.load_org_entities()
        self.saved_geocodes = json.load(open("./entity_recognition/saved-geocodes.json"))
    
    def load_geographies(self):
        # static data, states, towns, orgs in entity output format to filter out of geocoding results 
        self.load_state_entities()
        self.load_mass_town_entities()
    
    def load_state_entities(self):
        states = pd.read_csv("./entity_recognition/geo-data/states.csv")
        states_set = set()
        for idx in range(len(states)):
            tup = (states['state'][idx], 'LOC')
            states_set.add(tup)
        self.state_entities = states_set
    
    def load_mass_town_entities(self):
        towns = pd.read_csv("./entity_recognition/geo-data/mass-towns.csv")
        towns_set = set()
        for idx in range(len(towns)):
            tup = (towns['town'][idx], 'LOC')
            towns_set.add(tup)
        self.mass_town_entities = towns_set
    
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
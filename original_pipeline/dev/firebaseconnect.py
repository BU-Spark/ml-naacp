#firebaseconnect.py
#aidan gomez ethan springer sahanna kowshik DEC 2022

#handle reading and writing to firebase

import requests
#import encryption
import json

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
 
import firebase_secret

class firebase_pipeline:
    def __init__(self):
        self.url = firebase_secret.firebase_url
        self.serviceAccountPath = firebase_secret.firebase_service_acct
        self.dbName = firebase_secret.firebase_db_name #u'name'

        print("> Preparing Firestore access")
        try:
            self.cred = credentials.Certificate(self.serviceAccountPath)
            # Use a service account.
            self.app = firebase_admin.initialize_app(self.cred, {'databaseURL': self.url })
            self.db = firestore.client()
            self.db_ref = self.db.collection(self.dbName)
            self.parity = True
        except Exception as e:
            print('> Err: The Firestore configuration has failed. See exception: ')
            self.parity = False
            print(e)


    def read_firestore(self):
        print("> Reading Firestore...")
        try:
            docs = self.db_ref.stream()
            for doc in docs:
                print(f'{doc.id} => {doc.to_dict()}')
            return(docs)
         except Exception as e:
            print('> Err: The Firestore read has failed. See exception: ')
            print(e)

    def read_rtdb(self):
        res = requests.get(self.url)
        return(res)

    def write_rtdb(self, data):
        # ! assert <data> format here ! 
        # ! assert <data> originality here !
        #data = {"ea":ea, "ei":ei, "et":et}
        data = json.dumps(data)
        headers={'Content-Type': 'application/x-www-form-urlencoded',}
        res = requests.put(self.url, data=data, headers=headers)
        return(res)
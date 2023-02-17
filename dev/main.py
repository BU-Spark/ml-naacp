#main.py
#aidan gomez ethan springer sahanna kowshik DEC 2022

#entry point, pipelines RSS feed to models, manages, etc
#modified from EDA.ipynb in /pre-dev



#to-do: 
# classification generator [ ]
# firebase write [ ]

print("> main: loading libraries")
import pandas as pd
import numpy as np

import json
import spacy
import nltk 

#doc2vec

import rss_acq
import topicModel

#infer vector: [d2v_model_x.infer_vector(['the', 'election', 'results', 'will', 'be', 'returned', 'tomorrow'])])

class pipelining:
    def __init__(self):
        print("> main: class initializing")
        self.rss_handler = rss_acq.rss_acquisition()

    #takein rss feed
    def rss_in(self):
        self.rss_handler.rss_parse()
        data = self.rss_handler.acquiredToDF()
        #print(data.head())
        return(data)

    #write to firebase
    def firebase_out(self):
        pass

#model = Doc2Vec.load(fname)
class live_predictions:
    def __init__(self, doc2vec_model_path):
        #load doc2vec model from path
        self.engine = topicModel.internals(doc2vec_model_path)
        pass

    #entry point for running model on piece of input text
    def make(self, input_text):

        inferred_vector,similar_tags = self.engine.use_vectorizer(input_text)
        return(inferred_vector, similar_tags)
        pass
    
    

#define RSS & Firestore pipelines
a = pipelining()
#define topic engine
lp0 = live_predictions("./trainedmodels/20230215T185149")
#lp1 = live_predictions("./trainedmodels/xxxxxxxxxxxxxx")
data = a.rss_in()
if(data):
    lp0.make(data.iloc[0]['content'])


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
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

import rss_acq



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
    def __init__(self, model_path):
        #load doc2vec model from path
        self.model = Doc2Vec.load(model_path)
        pass

    #entry point for running model on piece of input text
    def make(self, input_text):

        inferred_vector = self.model.infer_vector(input_text.split(" "))
        print(inferred_vector)
        pass
    
    


a = pipelining()
data = a.rss_in()
b = live_predictions("./trainedmodels/november11_masked")
b.make(data.iloc[0]['content'])


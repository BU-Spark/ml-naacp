#main.py
#aidan gomez ethan springer sahanna kowshik DEC 2022

#entry point, pipelines RSS feed to models, manages, etc
#modified from EDA.ipynb in /pre-dev

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

class live_predictions:
    def __init__(self):
        print("> main: class initializing")
        self.rss_handler = rss_acq.rss_acquisition()
    def rss_in(self):
        self.rss_handler.rss_parse()
        data = self.rss_handler.acquiredToDF()
        print(data.head())
        #print(data)


a = live_predictions()

a.rss_in()

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

        #this returns a dataframe with columns:
        #'title','link','description','content','category','pubDate','UID'
        return(data)

    #write to firebase
    def firebase_out(self,df):
        #add integration w/ firebase connect. 
        #would do this but not sure how SE team wants data + what they want 
        #at this point 
        #should be straight forward though
        # - aidan
        pass

class live_predictions:
    def __init__(self, doc2vec_model_path, nn_path):
        #load doc2vec model from path
        self.engine = topicModel.internals(doc2vec_model_path, nn_path)
        pass

    #entry point for running model on piece of input text
    ### ORIGINAL FUNCTION ###
    def make(self, input_text):

        #inferred_vector,similar_tags = self.engine.use_vectorizer(input_text)
        inferred_vector,similar_tags,entities,label = self.engine.classify_plaintext(input_text)
        df_return = pd.DataFrame(columns=['vector', 'simlar_tags', 'entities', 'label'])
        df_return['vector'] = input_text
        df_return['similar_tags'] = pd.Series([similar_tags])
        df_return['entities'] = pd.Series([entities])
        df_return['label'] = label
        return(df_return) 

    ### TEST FUNCTION ###
    """ def make(self, input_text):
        #inferred_vector,similar_tags = self.engine.use_vectorizer(input_text)
        inferred_vector,similar_tags,entities,label = self.engine.classify_plaintext(input_text)
        print('predicted label:', label)
        df_return = pd.DataFrame(columns=['vector', 'simlar_tags', 'entities', 'label'])
        df_return['vector'] = input_text
        df_return['similar_tags'] = pd.Series([similar_tags])
        df_return['entities'] = pd.Series([entities])
        df_return['label'] = [label]
        return(df_return)
        pass """
    
#### ORIGINAL FUNCTION ##### 
def load_and_run():
#define RSS & Firestore pipelines
    a = pipelining()
    #define topic engine(s)
    lp0 = live_predictions("./trainedmodels/20230215T185149", "./trainedmodels/dvlabeler")
    lp1 = live_predictions("./trainedmodels/20230217T074231", "./trainedmodels/dvlabeler10000")
    lp2 = live_predictions("./trainedmodels/pens_model", "./trainedmodels/dvlabeler595")
    lp3 = live_predictions("./trainedmodels/pens_model_stratified", "./trainedmodels/dvlabeler561")
    #lp1 = live_predictions("./trainedmodels/xxxxxxxxxxxxxx")
    data = a.rss_in()
    if(isinstance(data, pd.DataFrame)):
        label_0 = []
        label_1 = []
        label_2 = []
        sim_tags_0 = []
        sim_tags_1 = []
        sim_tags_2 = []
        for index, row  in data.iterrows():
            guess_0 = lp0.make(row['content'])
            guess_1 = lp1.make(row['content'])
            guess_2 = lp2.make(row['content'])
            guess_3 = lp3.make(row['content'])
            #print(guess_0)
            #print(guess_1)
            file = open('./temp/runlog.txt','a')
            items = [row,guess_0,guess_1]
            label_0.append(guess_0['label'].iloc[0])
            label_1.append(guess_1['label'].iloc[0])
            label_2.append(guess_2['label'].iloc[0])
            label_3.append(guess_3['label'].iloc[0])
            sim_tags_0.append(guess_0['similar_tags'].iloc[0])
            sim_tags_1.append(guess_1['similar_tags'].iloc[0])
            sim_tags_2.append(guess_2['similar_tags'].iloc[0])
            sim_tags_3.append(guess_3['similar_tags'].iloc[0])
            for item in items:
	            file.write(str(item)+"\n")
            file.close()
            a.firebase_out([data,guess_0,guess_1])
        # data['label_0'] = label_0
        # data['similar_tags_0'] = sim_tags_0
        # data['label_1'] = label_1
        # data['similar_tags_1'] = sim_tags_1
        #data.to_csv('./temp/runlog.csv')
        guess_0.to_csv('./temp/model1_test_output.csv')
        guess_1.to_csv('./temp/model2_test_output.csv')
        guess_2.to_csv('./temp/model3_test_output.csv')
        guess_3.to_csv('./temp/model4_test_output.csv')


### TEST FUNCTION ###
""" def load_and_run():
#define RSS & Firestore pipelines
    a = pipelining()
    #define topic engine(s)
    lp0 = live_predictions("./trainedmodels/20230215T185149", "./trainedmodels/dvlabeler")
    lp1 = live_predictions("./trainedmodels/20230217T074231", "./trainedmodels/dvlabeler10000")
    #lp1 = live_predictions("./trainedmodels/xxxxxxxxxxxxxx")
    data = a.rss_in()
    if(isinstance(data, pd.DataFrame)):
        model1_output = pd.DataFrame(columns = ['vector', 'simlar_tags', 'entities', 'label', 'similar_tags'])
        model2_output = pd.DataFrame(columns = ['vector', 'simlar_tags', 'entities', 'label', 'similar_tags'])
        for index, row  in data.iterrows():
            guess_0 = lp0.make(row['content'])
            guess_1 = lp1.make(row['content'])

            file = open('./temp/runlog.txt','a')
            items = [row,guess_0,guess_1]

            for item in items:
	            file.write(str(item)+"\n")
            file.close()
            a.firebase_out([data,guess_0,guess_1])
            model1_output = pd.concat([model1_output, guess_0], ignore_index = True)
            model2_output = pd.concat([model2_output, guess_1], ignore_index = True)
        model1_output.to_csv('./temp/model1_test_output.csv')
        model2_output.to_csv('./temp/model2_test_output.csv') """


load_and_run()

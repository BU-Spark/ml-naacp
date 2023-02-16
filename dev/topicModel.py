import ast
import spacy
import multiprocessing
from sklearn.model_selection import train_test_split
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import json
import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
import numpy as np

import topicNetwork


class internals:
    def __init__(self, doc2vec_model_path):
        print("> topicModel: internals initializing")

        print("> topicModel: loading spacy dep. tagging model")
        self.nlp = spacy.load("en_core_web_trf")
        print("> topicModel: loaded spacy dep. tagging model")
        if(doc2vec_model_path == None):
            pass
        else:
            self.load_vectorizer(doc2vec_model_path)
        self.config()
        print("> topicModel: internals intialized")


    def load_vectorizer(self, doc2vec_model_path):
        print("> topicModel: loading model from path: ", doc2vec_model_path)
        self.doc2vec_model_path = doc2vec_model_path
        self.doc2vecmodel = Doc2Vec.load(self.doc2vec_model_path)
        print("> topicModel: loaded model from path: ", doc2vec_model_path)

    def use_vectorizer(self, article):
        if(self.doc2vecmodel):
            inferred_vector = self.doc2vecmodel.infer_vector(article.split(" "))
            similar = self.doc2vecmodel.dv.most_similar([inferred_vector]) 
        return(inferred_vector, similar)

    def use_vectorized_vectorizer(self, articles_column, core_boost=True):
       
        if(self.doc2vecmodel):

            ivv = np.vectorize(self.doc2vecmodel.infer_vector)
            sv = np.vectorize(self.doc2vecmodel.dv.most_similar)
            if(not core_boost):
                #this is broken!
                print("> topicModel: vectorized vector inference on size ", len(articles_column)) 
                ivc = ivv(articles_column)
                print("> topicModel: vectorized similar vectors on size ", len(articles_column)) 
                svc = sv(ivc)

            elif(core_boost):
                import multiprocessing
                import time
                print("> topicModel: multiprocessed vector inference on size ", len(articles_column))
                ivc = []
                with multiprocessing.pool.Pool(processes=10) as pool:
                    for result in pool.map(self.doc2vecmodel.infer_vector, articles_column):
                        #print(result)
                        ivc.append(result)
                print("> topicModel: multiprocessed similar vectors on size ", len(articles_column))
                svc = [] 
                with multiprocessing.pool.Pool(processes=10) as pool:
                    for result in pool.map(self.doc2vecmodel.dv.most_similar, ivc):
                        #print(result)
                        svc.append(result)
        return(ivc, svc)

    def config(self):
        #taggings which are NOT to be masked
        self.filtering_tags = ["nsubj", "ROOT", "dobj", "quantmod", "pobj"]
        self.test_size=0.2
        self.tag_mappings = json.load(open('tagconfig.json'))

        self.dropped_tags = json.load(open('droptags.json'))['drop']

        self.vector_size = 150
        self.min_count = 1
        self.doc2vec_epochs = 80

        self.mask_boolean=False
        #entity types with which a new label is automatically created in training

        self.config_entity_tagger = ['ORG', 
                                    'GPE', 
                                    'NORP', 
                                    'PERSON', 
                                    'LOC', 
                                    'EVENT',
                                    'FAC',
                                    'PRODUCT',
                                    'LANGUAGE',
                                    'LAW']

    def mask_article(self, article):
        try:
            doc = self.nlp(article)
            newstring = ""
            entities = []
            dependencies = []
            for ent in doc.ents:
                entities.append(ent.text)
            #print(entities)
            for token in doc:
                if(str(token.dep_) in self.filtering_tags):
                    dependencies.append(token.text)
            return(entities+dependencies)
        except Exception as inst:
            print("> topicModel: masking exception occurred")
            print("> topicModel: begin exception:")
            print(inst)
            print("> topicModel: end exception:")

    def get_entities(self, article):
        try:
            doc = self.nlp(article)
            newstring = ""
            entities = []
            dependencies = []
            for ent in doc.ents:
                if(ent.label_ in self.config_entity_tagger):
                    entities.append(ent.text)
            return(entities)
        except Exception as inst:
            print("> topicModel: entity extraction exception occurred")
            print("> topicModel: begin exception:")
            print(inst)
            print("> topicModel: end exception:")
    
    def train_doc2vec(self, train_corpus):
        print("> topicModel: training doc2vec model")
        print("> topicModel: doc2vec parameters: vector size: ", self.vector_size)
        print("> topicModel: doc2vec parameters: min_count: ", self.min_count)
        print("> topicModel: doc2vec parameters: epochs: ", self.doc2vec_epochs)

        d2v_model = Doc2Vec(vector_size=self.vector_size, min_count=self.min_count, epochs=self.doc2vec_epochs)
        #print(train_corpus)
        d2v_model.build_vocab(train_corpus)
        d2v_model.train(train_corpus, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)
        modelname = str(datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))

        d2v_model.save('./trainedmodels/'+modelname)
        print("> topicModel: doc2vec model: ", modelname, " finished training")
        return(d2v_model)

    def read_corpus_helper(self, pdf, tokens_only=False, entity_augment=True, verbose_train=True):
        print("> topicModel: doc2vec corpus processing beginning")
        cores = multiprocessing.cpu_count()
        it=0
        augment_count = 0
        exception_count = 0
        for article in pdf['body'].values:
            
            tagging = pdf.iloc[it]['tag']
            
                
            if(isinstance(article, str) and isinstance(tagging, str) and len(str(article))>0 and article != tagging):
                if(verbose_train):
                    print("----")
                    print(str(it), " of ", len(pdf['body'].values))
                if(self.mask_boolean==True):
                    article = " ".join(self.mask_article(article))
                article_entities = self.get_entities(article)
                if(verbose_train):
                    print("Article: ", article)
                    print("Tagging: ", tagging)
                    print("Entities: ", article_entities)
                    print("----")

                bad_sample = False
                try:
                    tagging = tagging.replace("'", '"')
                except Exception as inst:
                    exception_count +=1
                    tagging = "UNKNOWN"
                    bad_sample = True
                if(not bad_sample):    
                    tokens = gensim.utils.simple_preprocess(article)
                    if tokens_only:
                        yield tokens
                    else:
                        yield gensim.models.doc2vec.TaggedDocument(tokens, [tagging])
                        if(entity_augment==True):
                            for ae in article_entities:
                                yield gensim.models.doc2vec.TaggedDocument(tokens, [ae])
                                augment_count+=1
            else:
                pass
            it+=1
        if(exception_count > 0):
            print("> topicModel: Notice: Augmented ", augment_count, " samples building corpus")
        if(exception_count > 0):
            print("> topicModel: Warning: Discarded ", exception_count, " samples building corpus")
        
    def build_corpus(self, sample):
        print("> topicModel: building corpus")
        corpus = list(self.read_corpus_helper(sample, tokens_only=False))
        print("> topicModel: splitting train/test, test percentage: ", self.test_size)
        train_corpus, test_corpus = train_test_split(corpus, test_size = self.test_size)
        print("> topicModel: corpus ready ")
        print("> topicModel: corpus train size of: ", len(train_corpus), " samples")
        print("> topicModel: corpus test size of: ", len(test_corpus), " samples")
        return(train_corpus, test_corpus)

    def build_clean_sample(self, dataframe, body_field, tag_field):
        print("> topicModel: building training sample")
        dataframe = dataframe.rename(columns={body_field: "body", tag_field: "tag"})
        dataframe = dataframe[['body', 'tag']]
        dataframe['tag'] = dataframe['tag'].str.upper()
        return(dataframe)

    def correct_tags(self, sample):
        for key in self.tag_mappings.keys():
            #print(key)
            sample.loc[sample['tag'] == key] = self.tag_mappings[key]
            
        for key in self.dropped_tags:
            #print(key)
            sample = sample[sample.tag != key]
        return(sample)

    
def train():
    a = internals(None)
    df = pd.read_json('./data/News_Category_Dataset_v3.json', lines = True)
    train_sample_one = a.build_clean_sample(df, 'short_description', 'category')
    df2 = pd.read_csv ('./data/CNN_Articles.csv',on_bad_lines='skip')
    train_sample_two = a.build_clean_sample(df2, 'Article text', 'Category')
    df3 = pd.read_csv ('./data/CNN_Articles_2.csv',on_bad_lines='skip')
    train_sample_three = a.build_clean_sample(df3, 'Article text', 'Category')

    df4 = pd.read_csv('./data/news.tsv',on_bad_lines='skip', sep='\t')
    train_sample_four = a.build_clean_sample(df4, 'News body', 'Topic')

    #training_samples = pd.concat([train_sample_one,train_sample_two,train_sample_three])#, train_sample_four],ignore_index=True)
    #print(training_samples.tag.unique())
    training_samples = train_sample_four

    training_samples = a.correct_tags(training_samples)
    #print(training_samples.tag.unique())
    #train_sample_two = 
    training_corpus,testing_corpus = a.build_corpus(training_samples)
    model = a.train_doc2vec(training_corpus)


def save_corpus_to_pkl():
    import pickle
    a = internals(None)
    df4 = pd.read_csv('./data/news.tsv',on_bad_lines='skip', sep='\t')
    train_sample_four = a.build_clean_sample(df4, 'News body', 'Topic')


    training_samples = train_sample_four
    training_samples = a.correct_tags(training_samples)

    training_corpus,testing_corpus = a.build_corpus(training_samples)
    file = open('./data/corpus', 'wb')

    savedversion = training_corpus+testing_corpus
    pickle.dump(savedversion, file)
    file.close()

def load_corpus_from_pkl():
    import pickle
    file = open('./data/corpus', 'rb')
    object_file = pickle.load(file)
    return(object_file)
    file.close()



def test_vectorizer(path):
    a = internals(path)
    f = input()
    output = a.use_vectorizer(f)
    print(output)

a = internals('./trainedmodels/20230215T185149')
#save_corpus_to_pkl()
q = load_corpus_from_pkl()
q = np.array(q, dtype=object)
print(q.shape)
x = q[:,0]
y = q[:,1]

def vectorize_training_data():
    start = time.time()
    xmp = a.use_vectorized_vectorizer(x)
    import pickle
    file = open('./data/xmp', 'wb')
    savedversion = xmp
    pickle.dump(savedversion, file)
    file.close()
    end = time.time()
    print(end - start)


xmp = a.use_vectorized_vectorizer(x[0:100])
xmp = np.array(xmp, dtype=object)
print(xmp.shape)
print(x[0])
#print(len(xmp[0][0]))
print(xmp[1][0])
print(x.shape)
print(y.shape)

#print(q[1])
#train()
#test_vectorizer('./trainedmodels/20230215T185149')
#load_model()
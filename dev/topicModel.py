import ast
import numpy as np

import multiprocessing
from sklearn.model_selection import train_test_split

import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import spacy

from sklearn.ensemble import RandomForestClassifier
import string

from sentence_transformers import SentenceTransformer

#from sentence_transformers import SentenceTransformer?
#model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
#embeddings = model.encode(sentences)

import json
import pandas as pd

import datetime
import time

import matplotlib.pyplot as plt


import topicNetwork

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class internals:
    def __init__(self, doc2vec_model_path, nn_path):
        print("> topicModel: internals initializing")

        print("> topicModel: loading spacy dep. tagging model")
        self.nlp = spacy.load("en_core_web_trf")
        print("> topicModel: loaded spacy dep. tagging model")
        if(doc2vec_model_path == None):
            pass
        else:
            self.load_vectorizer(doc2vec_model_path)
        if(nn_path==None):
            pass
        else:
            self.network = topicNetwork.network_handler(nn_path)
        self.randomforest = None
        self.config()
        print("> topicModel: internals intialized")


    def load_vectorizer(self, doc2vec_model_path):
        print("> topicModel: loading model from path: ", doc2vec_model_path)
        self.doc2vec_model_path = doc2vec_model_path
        self.doc2vecmodel = Doc2Vec.load(self.doc2vec_model_path)
        print("> topicModel: loaded model from path: ", doc2vec_model_path)
    def fit_random_forest(self, X, y):
        clf = RandomForestClassifier(max_depth=70, random_state=0)
        self.random_forest = clf.fit(X, y)
        pass
    def predict_random_forest(self, x):
        return self.random_forest.predict([x])

    def classify_plaintext(self, article):
        print("> topicModel: nn inference")
        print(article)
        v,s = self.use_vectorizer(" ".join(article))
        e = self.get_entities(article)
        vec = v
        vec=np.array([np.array(vec, dtype=np.float64)], dtype=np.float64)
        label = self.network.classify(vec)
        print("> topicModel: nn inference complete")
        return(v,s,e,self.tag_list[label])
    def use_vectorizer(self, article):
        if(self.doc2vecmodel):
            inferred_vector = self.doc2vecmodel.infer_vector(article.split(" "))
            similar = self.doc2vecmodel.dv.most_similar([inferred_vector]) 
        return(inferred_vector, similar)

    def use_vectorized_vectorizer(self, articles_column, core_boost="SLOW"):
        #a vectorized implementation of the vectorizer
        #ie, applying the doc2vec model, which vectorizes text,
        #in a vectorized manner
        #for the purpose of speed
        #confusing
        if(self.doc2vecmodel):

            ivv = np.vectorize(self.doc2vecmodel.infer_vector)
            sv = np.vectorize(self.doc2vecmodel.dv.most_similar)
            if(not core_boost):
                #this is broken!
                print("> topicModel: vectorized vector inference on size ", len(articles_column)) 
                ivc = ivv(articles_column, epochs=200)
                print("> topicModel: vectorized similar vectors on size ", len(articles_column)) 
                svc = sv(ivc)
            elif(core_boost=="SLOW"):
                ivc = []
                vectlam = lambda s,a,b : s.doc2vecmodel.infer_vector(doc_words=a, epochs=b)
                args = [(self,articles_column[i], 200) for i in range(len(articles_column))]
                progress = 0
                total = len(args)
                
                import multiprocessing
                '''
                with multiprocessing.pool.Pool(processes=10) as pool:
                    for result in pool.map(vectlam, args):
                        #print(result)
                        ivc.append(result)
                '''
                for x in args:
                    print(str((progress/total)* 100), " %   \r",end='')
                    #print(x)
                    vector = vectlam(x[0], x[1],x[2])
                    ivc.append(vectlam(x[0], x[1],x[2]))
                    progress+=1
                    #print(vector)
                print("")
                svc = []
                with multiprocessing.pool.Pool(processes=10) as pool:
                    for result in pool.map(self.doc2vecmodel.dv.most_similar, ivc):
                        #print(result)
                        svc.append(result)
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

        self.vector_size = 300
        self.min_count = 1
        self.doc2vec_epochs = 200
        self.workers=14
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
        tag_list = []
        for key in self.tag_mappings:
            tag_list.append(self.tag_mappings[key])
        self.tag_list = list(np.unique(np.array(tag_list)))
        print("Tag List: ")
        print(self.tag_list)
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

        d2v_model = Doc2Vec(vector_size=self.vector_size, min_count=self.min_count,workers=self.workers, epochs=self.doc2vec_epochs)#, workers=10)
        #print(train_corpus)
        print("> topicModel: building vocab")
        d2v_model.build_vocab(train_corpus)
        print("> topicModel: training, samples = ", str(d2v_model.corpus_count))
        d2v_model.train(train_corpus, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)
        # modelname = str(datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
        modelname = 'pens_new'

        d2v_model.save('./trainedmodels/'+modelname)
        print("> topicModel: doc2vec model: ", modelname, " finished training")
        return(d2v_model)

    def embedding_corpus_transformer(self, pdf,verbose_train=True, entity_augment=False):
        corpus_transformer = []
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        print("> topicModel: sentenceTransformer corpus processing beginning")
        cores = multiprocessing.cpu_count()
        english_stopwords = stopwords.words('english')
        punctuation = list(string.punctuation)
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

                #tokenize, prepare article, remove stopwords

                atokens = word_tokenize(article.lower())
                atokens=[word.lower() for word in atokens if word.isalpha()]
                atokens_wo_stopwords = [t for t in atokens if t not in english_stopwords and t not in punctuation]
                article = " ".join(atokens_wo_stopwords)

                
                if(entity_augment==True):
                    article_entities = list(set(self.get_entities(article)))
                else:
                    article_entities = []
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
                    embeddings = model.encode([article])
                    corpus_transformer.append([article, tagging, article_entities, embeddings])
            it+=1
        return(corpus_transformer)

    def read_corpus_helper(self, pdf, tokens_only=False, entity_augment=False, verbose_train=True):
        print("> topicModel: doc2vec corpus processing beginning")
        cores = multiprocessing.cpu_count()
        english_stopwords = stopwords.words('english')
        punctuation = list(string.punctuation)
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

                #tokenize, prepare article, remove stopwords

                atokens = word_tokenize(article.lower())
                atokens=[word.lower() for word in atokens if word.isalpha()]
                atokens_wo_stopwords = [t for t in atokens if t not in english_stopwords and t not in punctuation]
                article = " ".join(atokens_wo_stopwords)

                
                if(entity_augment==True):
                    article_entities = list(set(self.get_entities(article)))
                else:
                    article_entities = []
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
    
    def train_bert_vectorizer(self, sample):
        #model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        #embeddings = model.encode(sample)

        pass
        
    
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
    a = internals(None,None)
    df = pd.read_json('./data/News_Category_Dataset_v3.json', lines = True)
    train_sample_one = a.build_clean_sample(df, 'short_description', 'category')
    df2 = pd.read_csv ('./data/CNN_Articles.csv',on_bad_lines='skip')
    train_sample_two = a.build_clean_sample(df2, 'Article text', 'Category')
    df3 = pd.read_csv ('./data/CNN_Articles_2.csv',on_bad_lines='skip')
    train_sample_three = a.build_clean_sample(df3, 'Article text', 'Category')

    df4 = pd.read_csv('./data/news.tsv',on_bad_lines='skip', sep='\t')
    train_sample_four = a.build_clean_sample(df4, 'News body', 'Topic')
    df5 = pd.read_csv('./data/NewsCategorizer.csv',on_bad_lines='skip')
    train_sample_five = a.build_clean_sample(df, 'short_description', 'category')
    #training_samples = pd.concat([train_sample_one,train_sample_two,train_sample_three])#, train_sample_four],ignore_index=True)
    #print(training_samples.tag.unique())
    training_samples = pd.concat([train_sample_one,train_sample_two,train_sample_three,train_sample_four,train_sample_five], ignore_index=True)
    #training_samples = pd.concat([train_sample_two,train_sample_three], ignore_index=True)
    training_samples = a.correct_tags(training_samples)
    #print(training_samples.tag.unique())
    #train_sample_two = 
    import pickle
    training_corpus,testing_corpus = a.build_corpus(training_samples)
    file = open('./data/corpus_b', 'wb')
    savedversion = training_corpus+testing_corpus
    pickle.dump(savedversion, file)
    file.close()
    model = a.train_doc2vec(training_corpus)


def save_corpus_to_pkl():
    import pickle
    a = internals(None,None)
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
    file = open('./data/corpus_b', 'rb')
    object_file = pickle.load(file)
    return(object_file)
    file.close()




def test_vectorizer(path):
    a = internals(path,None)
    f = input()
    output = a.use_vectorizer(f)
    print(output)


def train_neural_network(vectorizermodel):
    #init internals and load corpus
    #'./trainedmodels/20230420T194401'
    a = internals(vectorizermodel,None)
    q = load_corpus_from_pkl()
    print("> topicModel: training neural network using doc2vec model ", vectorizermodel)
    q = np.array(q, dtype=object)
    print(q.shape)
    #x is vector portion of corpus
    x = q[:,0]
    #y is label
    y = q[:,1]
    p_x = []
    p_y = []
    example_count = 0
    indexing = 0
    #while there are fewer than n examples and we remain within the length of the vectors
    print("> topicModel: creating training set for neural network ")
    while example_count < len(x)-100 and indexing<len(x):
        if(y[indexing][0] in a.tag_list and y[indexing][0]!="NEWS"):
            #iterating through the corpus
            #append current article
            #append current label
            p_x.append(x[indexing])
            p_y.append(a.tag_list.index(y[indexing][0]))
            #print(x[indexing], a.tag_list.index(y[indexing][0]))
            example_count+=1
        else:
            
            pass
        indexing+=1
    print("> topicModel: text training set of ", str(example_count),  " created for neural network ")
    #apply vectorizer to p_x
    #xmp is inferred vectors, smp is the similar vectors
    print("> topicModel: text training set of ", str(example_count),  " created for neural network ")
    print("> topicModel: applying doc2vec model to text training set")
    xmp,smp = a.use_vectorized_vectorizer(p_x)

    #combine vectors and labels in list
    data = list(zip(xmp, p_y))
    #train, test

    train, test = train_test_split(data, test_size = a.test_size)
    
    train=np.array([np.array(xi) for xi in train])
    test=np.array([np.array(xi) for xi in test])
    
    print(len(a.tag_list))
    X_train=np.array([np.array(xi, dtype=np.float64) for xi in train[:,0]])
    Y_train=np.array([np.array(np.array(xi, dtype=np.float64)) for xi in train[:,1]])

    X_test=np.array([np.array(xi, dtype=np.float64) for xi in test[:,0]])
    Y_test=np.array([np.array(np.array(xi, dtype=np.float64)) for xi in test[:,1]])

    topicNetwork.train_network(X_train,Y_train,X_test,Y_test)

        
#@train_neural_network()
def test_network():
    a = internals('./trainedmodels/20230421T122349',None)
    q = load_corpus_from_pkl()
    network = topicNetwork.network_handler('./trainedmodels/dvlabeler10000')
    print(len(q))
    for x in range(0, 10):
        print(" ".join(q[x][0]))
        v,s = a.use_vectorizer(" ".join(q[x][0]))
        vec = v
        vec=np.array([np.array(vec, dtype=np.float64)], dtype=np.float64)
        label = network.classify(vec)
        print(a.tag_list[label])
        print(s)


def train_random_forest():
    a = internals('./trainedmodels/20230217T074231',None)
    q = load_corpus_from_pkl()

    q = np.array(q, dtype=object)
    print(q.shape)
    x = q[:,0]
    y = q[:,1]
    p_x = []
    p_y = []
    example_count = 0
    indexing = 0
    while example_count < len(x) and indexing<len(x):
        if(y[indexing][0] in a.tag_list and y[indexing][0]!="NEWS"):
            p_x.append(x[indexing])
            p_y.append(a.tag_list.index(y[indexing][0]))
            #print(x[indexing], a.tag_list.index(y[indexing][0]))
            example_count+=1
        else:
            
            pass
        indexing+=1
    xmp,smp = a.use_vectorized_vectorizer(p_x)
    data = list(zip(xmp, p_y))
    train, test = train_test_split(data, test_size = a.test_size)
    
    train=np.array([np.array(xi) for xi in train])
    test=np.array([np.array(xi) for xi in test])

    print(len(a.tag_list))
    X_train=np.array([np.array(xi, dtype=np.float64) for xi in train[:,0]])
    Y_train=np.array([np.array(np.array(xi, dtype=np.float64)) for xi in train[:,1]])

    X_test=np.array([np.array(xi, dtype=np.float64) for xi in test[:,0]])
    Y_test=np.array([np.array(np.array(xi, dtype=np.float64)) for xi in test[:,1]])

    a.fit_random_forest(X_train,Y_train)
    right = 0
    total = 0
    for x in range(0, len(X_test)):
        y = a.predict_random_forest(X_test[x])
        if(y == Y_test[x]):
            right+=1
        total+=1
    print(str(right), " of ", str(total))

def embeddings_with_transformer():
    a = internals(None,None)
    df = pd.read_json('./data/News_Category_Dataset_v3.json', lines = True)
    train_sample_one = a.build_clean_sample(df, 'short_description', 'category')
    df2 = pd.read_csv ('./data/CNN_Articles.csv',on_bad_lines='skip')
    train_sample_two = a.build_clean_sample(df2, 'Article text', 'Category')
    df3 = pd.read_csv ('./data/CNN_Articles_2.csv',on_bad_lines='skip')
    train_sample_three = a.build_clean_sample(df3, 'Article text', 'Category')

    df4 = pd.read_csv('./data/news.tsv',on_bad_lines='skip', sep='\t')
    train_sample_four = a.build_clean_sample(df4, 'News body', 'Topic')
    df5 = pd.read_csv('./data/NewsCategorizer.csv',on_bad_lines='skip')
    train_sample_five = a.build_clean_sample(df, 'short_description', 'category')
    #training_samples = pd.concat([train_sample_one,train_sample_two,train_sample_three])#, train_sample_four],ignore_index=True)
    #print(training_samples.tag.unique())
    training_samples = pd.concat([train_sample_one,train_sample_two,train_sample_three,train_sample_four,train_sample_five], ignore_index=True)
    #training_samples = pd.concat([train_sample_two,train_sample_three], ignore_index=True)
    training_samples = a.correct_tags(training_samples)
    #print(training_samples.tag.unique())
    #train_sample_two = 
    import pickle
    transformer_corpus = a.embedding_corpus_transformer(training_samples)
    file = open('./data/transformer_corpus', 'wb')
    savedversion = transformer_corpus
    pickle.dump(savedversion, file)
    file.close()
    #model = a.train_doc2vec(training_corpus)
def multiprocessed_embeddings_with_transformer():
    transformer_corpus = []
    a = internals(None,None)
    df = pd.read_json('./data/News_Category_Dataset_v3.json', lines = True)
    train_sample_one = a.build_clean_sample(df, 'short_description', 'category')
    df2 = pd.read_csv ('./data/CNN_Articles.csv',on_bad_lines='skip')
    train_sample_two = a.build_clean_sample(df2, 'Article text', 'Category')
    df3 = pd.read_csv ('./data/CNN_Articles_2.csv',on_bad_lines='skip')
    train_sample_three = a.build_clean_sample(df3, 'Article text', 'Category')

    df4 = pd.read_csv('./data/news.tsv',on_bad_lines='skip', sep='\t')
    train_sample_four = a.build_clean_sample(df4, 'News body', 'Topic')
    df5 = pd.read_csv('./data/NewsCategorizer.csv',on_bad_lines='skip')
    train_sample_five = a.build_clean_sample(df, 'short_description', 'category')
    #training_samples = pd.concat([train_sample_one,train_sample_two,train_sample_three])#, train_sample_four],ignore_index=True)
    #print(training_samples.tag.unique())
    training_samples = pd.concat([train_sample_one,train_sample_two,train_sample_three,train_sample_four,train_sample_five], ignore_index=True)
    #training_samples = pd.concat([train_sample_two,train_sample_three], ignore_index=True)
    training_samples = a.correct_tags(training_samples)
    #print(training_samples.tag.unique())
    #train_sample_two = 
    import pickle
    with multiprocessing.pool.Pool(processes=10) as pool:
        for result in pool.map(a.embedding_corpus_transformer, training_samples):
            print(result)
            transformer_corpus.append(result)
    file = open('./data/transformer_corpus', 'wb')
    savedversion = transformer_corpus
    pickle.dump(savedversion, file)
    file.close()
#train_random_forest()
#train()
#multiprocessed_embeddings_with_transformer()
#embeddings_with_transformer()
train_neural_network('./trainedmodels/20230421T134110')
#test_network()
#train()
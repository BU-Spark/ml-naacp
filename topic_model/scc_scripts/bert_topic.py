import pandas as pd
import dask.dataframe as dd 

from bertopic import BERTopic

from sentence_transformers import SentenceTransformer 
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer



lndf = dd.read_csv("/projectnb/sparkgrp/ds-naacp-media-bias/Spring-2022/TBG_unique_raw.csv", blocksize=100e6, dtype={'author': 'object',
                                                                                         'edition': 'object',
                                                                                         'licensor_indexing_terms': 'object'})
print("get sample of boston globe data...")
sample = lndf.sample(frac=0.1, random_state=1)
sample_pd = sample.compute()

print("saving sample to csv...")
sample_pd.to_csv("bglobe_100k_sample.csv") # saved here on scc: /projectnb/sparkgrp/mvoong

print("cleaning article body text...")
new_body = []
for body in sample["body"]:
    if body == "body":
        new_body.append("")
    else:
        try:
            new_body.append(body.split("body ")[1])
        except IndexError:
            new_body.append(body)
        except AttributeError:
            new_body.append("")
clean_body=pd.Series(new_body)

print("initializing embeddings...")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
vectorizer_model = CountVectorizer(stop_words='english')
representation_model = MaximalMarginalRelevance(diversity=0.8)
embeddings_body = sentence_model.encode(clean_body)
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

print("fitting topic model...")
bglobe_topicmodel = BERTopic(n_gram_range = (1,3),
                          embedding_model=sentence_model,
                          vectorizer_model=vectorizer_model,
                          representation_model=representation_model,
                          ctfidf_model=ctfidf_model)
topic_lede, prob_lede = bglobe_topicmodel.fit_transform(clean_body, embeddings_body)

print("save topic model...")
bglobe_topicmodel.save("bglobe_519_body") # saved here on scc: /projectnb/sparkgrp/mvoong
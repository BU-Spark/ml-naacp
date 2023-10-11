import pandas as pd
import dask.dataframe as dd 
import openai

from bertopic import BERTopic
from bertopic.representation import OpenAI
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer 
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer

import random

openai.api_key = # GitHub version does not include the OpenAI API key, please replace with your own key

lndf = dd.read_csv("/projectnb/sparkgrp/ds-naacp-media-bias/Spring-2022/TBG_unique_raw.csv", blocksize=100e6, dtype={'author': 'object',
                                                                                         'edition': 'object',
                                                                                         'licensor_indexing_terms': 'object'})
print("get sample of boston globe data...")
sample = lndf.sample(frac=0.1, random_state=1)
sample_pd = sample.compute()

print("saving sample to csv...")
sample_pd.to_csv("../datasets/bglobe_100k_sample.csv") # saved here on scc: /projectnb/sparkgrp/dyxu/datasets/bglobe_100k_sample.csv

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
hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=1)
# representation_model = KeyBERTInspired()
# representation_model = OpenAI(model="gpt-3.5-turbo", chat=True, exponential_backoff=True, delay_in_seconds=random.randint(6,60))
representation_model = MaximalMarginalRelevance(diversity=0.8)
embeddings_body = sentence_model.encode(clean_body)
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

print("fitting topic model...")
bglobe_topicmodel = BERTopic(n_gram_range = (1,3), top_n_words=15,
                          embedding_model=sentence_model,
                          vectorizer_model=vectorizer_model,
                          umap_model=umap_model,
                          hdbscan_model=hdbscan_model,
                          representation_model=representation_model,
                          ctfidf_model=ctfidf_model)
topic, prob = bglobe_topicmodel.fit_transform(clean_body, embeddings_body)

bglobe_topicmodel.reduce_topics(clean_body, nr_topics=350)
print("save topic model...")
bglobe_topicmodel.save("../models/bglobe_519_body_350") # saved here on scc: /projectnb/sparkgrp/dyxu/models/bglobe_519_body_350
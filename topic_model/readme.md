## NAACP Media Bias Project -- Roadmap since 2023 Summer
-- *Maintained by Michelle Voong (mvoong@bu.edu) and Dingyuan Xu (dyxu@bu.edu).* 

### Foreword
This file describes past attempts that we have made for building the topic modelling pipeline for the NAACP Media Bias Project. Overall, our clients want to see an automated tool that tags news articles with a topic and a location, so that they could compare topic distributions of different neighborhoods and identify any potential disparities. This readme describes only the topic modelling part, and is separated by major changes in pipeline architecture. 

### BERTopic + ChatGPT Approach (2023 Summer -- 2023 Winter)
This early approach employs an unsupervised clustering method and label each cluster with a bag of words representation. Documentation of BERTopic can be found [here](https://maartengr.github.io/BERTopic/index.html "here"). Then, ChatGPT is asked to come up with a comprehensive topic befitting the bag of words, so that topics are more intuitively represented. 
![BERTopic Training Flowchart](https://github.com/BU-Spark/ml-naacp/blob/pure_embedding_approach/topic_model/images/bertopic_training.png?raw=true)
During inference, each news article gets labeled with a specific cluster, along with ChatGPT's label.
![BERTopic Inferencing Flowchart](https://github.com/BU-Spark/ml-naacp/blob/pure_embedding_approach/topic_model/images/bertopic_inference.png?raw=true)

Scripts for this approach are located in `topic_model/summer_edits/inference.ipynb` and the BERTopic model's training script is `bertopic.py` under the same directory. You are also welcome to explore other helper scripts for various file format transition in order to have the results exported to the Web App, as well as experiments conducted in `topic_model_test.ipynb`. 

To better restrict ChatGPT's hallucination and random behaviors when generating natural language topics out of a bag of words, we found a news taxonomy (file can be found at `topic_model/prompt_with_taxonomy/Content_Taxonomy.csv`) which provides fine-grained topics for all news categories. Initial attempts were made to use RAG capabilities of GPT and directly prompting it with the taxonomy, but none improved the two issues significantly, which led to the second word embedding based approach. 
### Word Embedding-based Approach (2024 --)
Instead of prompting GPT, we called OpenAI's word embedding model ([Documentation](http://https://platform.openai.com/docs/guides/embeddings "Documentation")) to get embeddings for all articles and labels respectively, and find the most similar label to every news article. This approach has both faster runtime and lower token cost, and it has replaced training local models with calling external APIs, which reduces server load. 
![Pure Embedding Approach Flowchart](https://github.com/BU-Spark/ml-naacp/blob/pure_embedding_approach/topic_model/images/pure_embedding.png?raw=true)
Inference script for this pipeline can be found in `topic_model/pure_embedding/inference_pure_embedding.ipynb`. Similarly, experiments as well as analysis can be found in `pure_embedding_approach.ipynb` and `pure_embedding_analysis.ipynb`. 

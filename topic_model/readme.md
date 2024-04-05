## NAACP Media Bias Project -- Roadmap since 2023 Summer
-- *Maintained by Michelle Voong (mvoong@bu.edu) and Dingyuan Xu (dyxu@bu.edu).* 

### Foreword
This file describes past attempts that we have made for building the topic modelling pipeline for the NAACP Media Bias Project. Overall, our clients want to see an automated tool that tags news articles with a topic and a location, so that they could compare topic distributions of different neighborhoods and identify any potential disparities. This readme describes only the topic modelling part, and is separated by major changes in pipeline architecture. 

### BERTopic + ChatGPT Approach (2023 Summer -- 2023 Winter)
This early approach employs an unsupervised clustering method and label each cluster with a bag of words representation. Documentation of BERTopic can be found [here](https://maartengr.github.io/BERTopic/index.html "here"). Then, ChatGPT is asked to come up with a comprehensive topic befitting the bag of words, so that topics are more intuitively represented. 
*TODO: Add image*
During inference, each news article gets labeled with a specific cluster, along with ChatGPT's label.
*TODO: Add image*

Scripts for this approach are located in `topic_model/summer_edits/inference.ipynb` and the BERTopic model's training script is `bertopic.py` under the same directory. You are also welcome to explore other helper scripts for various file format transition in order to have the results exported to the Web App, as well as experiments conducted in `topic_model_test.ipynb`. 

To better restrict ChatGPT's hallucination and random behaviors when generating natural language topics out of a bag of words, we found a news taxonomy which provides fine-grained topics for all news categories. 
### Word Embedding-based Approach (2024 --)


```code
import numpy as np

```
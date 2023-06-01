# NAACP Media Bias Project
This repository contains a software pipeline to output topics and geocoded locations mentioned in an article. 

# Overview

## Project Description
This project involves building on a proof of concept for evaluating media coverage of Black Americans and other demographics in Boston including: coverage of predominantly Black neighborhoods through topic modeling; coverage of explicit mention of race through topic modeling and frames; extraction of quotes to verify demographics of sources.

This data will be presented through a web interface tool for editors that shows a dashboard of coverage and allows for deeper analysis of underlying data.

## Getting Started 

### `./entity_recognition`
The `entity_recognition` folder contains the code to extract mentions of locations in articles. Entity recognition is used to identify locations mentioned in an article, geocode the locations, and if the location is in Boston, return the census tract demographics of the mentioned location.

### `./gbh_rss`
The `EDA` folder contains an exploritory data analysis of the dataset. The `EDA.ipynb` file contains the code to generate the EDA. The `EDA_Notebook_Spring_2023.ipynb` file contains the latest output of the EDA.

### `./topic_model`
The `topic_model` folder contains a BERTopic Model to perform topic model to perform topic modelling on articles. 

### `./original_pipeline`
The `original_pipeline` folder contains a doc2vec model (trained on different news data sets) to perform topic modelling on articles. Given an article the model will output entities and news section the article belongs to.  


## Resources

### Data Sets

* LexisNexis Dataset: in sparkgrp project on SCC. `/projectnb/sparkgrp/ds-naacp-media-bias/TBG_unique_raw.csv` 
* WGBH RSS Feed: https://www.wgbh.org/news/newsfeed.rss
* Boston Census Tract Demographics: https://data.boston.gov/dataset/2020-census-for-boston/resource/013aba13-5985-4067-bba4-a8d3ca9a34ac

### References

1. BERTopic: https://maartengr.github.io/BERTopic/index.html
2. BERT NER: https://huggingface.co/dslim/bert-large-NER

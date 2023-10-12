### 2023 Fall: Prompt GPT with a list of taxonomy indices and have it choose the best topic label for each BERTopic cluster
1. Open prompt_gpt_with_taxonomy.ipynb, run through cells (TODO: code cleanup) to generate taxonomy list
  <br>1.1. Currently using tier 2 in the taxonomy file found, level of granularity seems appropriate
  <br>1.2. 165/230 topics were "made up" by GPT, but GPT-generated topics better describe the clusters after prompting with taxonomy list
2. To inference the new pipeline, run the inference_taxonomy.ipynb in this directory
3. New training script that resulted in 230 BERTopic clusters is bert_topic_experiments.py.

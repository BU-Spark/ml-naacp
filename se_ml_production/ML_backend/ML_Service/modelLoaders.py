# BERT NER
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

# BERT TOPIC
from bertopic import BERTopic 

def load_bert_NER():
	"""
	Huggingface implementation of BERT NER LLM
	"""
	tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
	model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
	nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="max")

	return nlp


def load_bert_TOPIC():
	"""
	In house trained BERT TOPIC Model
	"""
	topic_model = BERTopic.load("./llm_models/BERTopic_CPU_M1") # BERTopic model dir
	return topic_model


# BERT NER
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

# BERT TOPIC
from bertopic import BERTopic 
from sentence_transformers import SentenceTransformer 

def load_bert_NER():
	"""
	Huggingface implementation of BERT NER LLM
	"""
	tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
	model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
	nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="max")

	return nlp


def load_bert_TOPIC():
	#loading bert Topic model
	topic_model = BERTopic.load(
		"./llm_models/bglobe_519_body_230_cereal", 
		embedding_model=SentenceTransformer("all-MiniLM-L6-v2")
	)
	return topic_model


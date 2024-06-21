# Llama 7B
from langchain.prompts import PromptTemplate # Prompt
from langchain_community.llms import LlamaCpp # Llama CPP
from langchain.callbacks.manager import CallbackManager
from langchain_core.output_parsers import StrOutputParser # Parser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Spanmarker NER
import spacy
from span_marker import SpanMarkerModel

def load_spanmarker_NER():
	"""
	SpanMarker is a framework for training powerful Named Entity Recognition models using familiar encoders such as BERT, RoBERTa and ELECTRA.
	IOB, IOB2, BIOES, BILOU Scheme Compatible.
	"""
	nlp = spacy.load("en_core_web_sm", exclude=["ner"])
	nlp.add_pipe("span_marker", config={"model": "tomaarsen/span-marker-roberta-large-ontonotes5"})
	return nlp

def load_llama_7B():
	"""
	Llama 2.0 trained on 7B parameters, medium model. Created by Meta.
	"""
	llama_model_path = "./llm_models/llama-2-7b-chat.Q4_K_M.gguf"

	prompt = PromptTemplate(
		input_variables=["headline"],
		template="""<<SYS>> \n You are an assistant tasked in geo-locating \
		this news article. \n <</SYS>> \n\n [INST] Generate a SHORT response \
		of where you think this article is talking about. BE SPECIFIC AS POSSIBLE. IT IS IMPERATIVE THAT YOU HIGHLIGHT THE MOST SPECIFIC LOCATION. Give your response in the following format: \
		1.Y/N indicating whether the article is talking about a region of Boston. \n 2.The specific location within the city you got if you got Y in the first question. \
		3. The involved specific locations or organizations EXPLICITLY FOUND WITHIN THE ARTICLE that influenced your decision. \
		If you do not know, PLEASE GIVE THE BEST GUESS AS POSSIBLE. \n\n
		Headline: \n\n {headline} \n\n [/INST]""",
	)

	llm = LlamaCpp(
		model_path=llama_model_path,
		n_gpu_layers=1,
		n_batch=1024,
		n_ctx=2048,
		f16_kv=True,
		callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
		verbose=True,
	)

	output_parser = StrOutputParser()
	chain = prompt | llm | output_parser

	return chain
# Llama 3.1 8B
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

# Load Llama 3.1 8B
def load_llama_3_1():
	"""
	Llama 3.1 trained on 8B parameters, medium model (4 Quant), Instruct type. Created by Meta.
	"""
	llama_model_path = "./llm_models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

	prompt = PromptTemplate(
		input_variables=["headline", "body"],
		template="""
		<|begin_of_text|><|start_header_id|>system<|end_header_id|>
		
		Cutting Knowledge Date: December 2023
		Today Date: 26 Jul 2024

		You are an expert in geo-location and have a deep understanding of specific places and organizations. In a short response, your task is to identify and provide the most exact real location mentioned in the news article. This can be a place, organization, facility, or any location that can help identify where the article takes place or talks about. Also, mention any specific locations or organizations explicitly found within the article that influenced your decision. If you cannot determine a location, state that explicitly. Do not discuss anthing else.<|eot_id|><|start_header_id|>user<|end_header_id|>

		You are tasked in geo-locating this news article. Generate a SHORT response specifying the most exact location you can find mentioned in the article. Give your answer in the following format:
		1. If there is one, the city the article is talking about. Otherwise, state that it can't be located. 
		2. The specific place within the city you got if you found one.
		3. The involved specific locations or organizations EXPLICITLY FOUND WITHIN THE ARTICLE that influenced your decision.
		
		If you cannot determine a location, state that explicitly. DO NOT MAKE UP INFORMATION. This is the news article:
		Headline: \n\n {headline} \n\n Body: \n\n {body} \n\n <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
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
import secret
import pandas as pd
from io import StringIO  # Import StringIO
from fastapi import UploadFile # For typing
from collections import Counter
from fastapi import UploadFile
from global_state import global_instance

def is_duplicate_article(tag, articles_collection):
	queryArticles = {
		'$and': [
			{'userID': global_instance.get_data("userID")},
			{'content_id': tag}
		]
	}
	return articles_collection.find_one(queryArticles) is not None

def run_validation(client, df):
	db_prod = client[secret.db_name]
	collection_list = db_prod.list_collection_names()

	if ('articles_data' in collection_list):
		articles_collection = db_prod['articles_data']
		df['is_duplicate'] = df['content_id'].apply(lambda tag: is_duplicate_article(tag, articles_collection))
		print(df[df['is_duplicate'] == False]['content_id'])
		df = df.drop(df[df['is_duplicate']].index).drop(columns='is_duplicate')

		if (df.empty):
			print("[WARNING] Dataset is a subset of existing articles in MongoDB! Removing duplications left 0 articles to process.")
	return df

async def read_csv(file: UploadFile) -> pd.DataFrame:
	"""
	Read in a file type. Converts it into a stream of bytes which then is fed into a pandas df reader.
	---
	Parameters:
	file: A UploadFile object that allows for read operations
	---
	Returns: A pandas dataframe
	"""
	try:
		if (not file):
			raise Exception(f"Error! Passed in an empty file!")
		file_contents = await file.read()
		file_content_str = file_contents.decode("utf-8")
		df = pd.read_csv(StringIO(file_content_str), low_memory=False) # Use StringIO to convert the string data to a file-like object so it can be read into a pandas dataframe

		cols_to_drop = [col for col in df.columns if col.startswith('Unnamed')] # Weird columns if you do some conversions
		df = df.drop(columns=cols_to_drop)
	except Exception as e:
		raise Exception(f"Could not read and convert csv file to pandas df. Error: {e}")
	return df

def validate_csv(df: pd.DataFrame) -> bool:
	db_manager = global_instance.get_data("db_manager")
	
	data_col_schema = ['Headline','Publisher','Byline','Paths','Publish Date','Body', 'content_id']

	try:
		current_df_col = list(df.columns)
		print(current_df_col)
		if (not Counter(current_df_col) == Counter(data_col_schema)):
			raise Exception(f"CSV Columns do not match the required data schema!")

		# Check for duplicates
		df = db_manager.run_job(
				run_validation, 
				db_manager.act_con[0]['connection'], # Argument 1 (1st connection)
				df,
				connection_obj=db_manager.act_con[0]
			)
		print("[INFO] CSV Validation Complete.")
		return df
	except Exception as e:
		raise Exception(f"[WARNING] CSV Validation failed! {e}")
	return
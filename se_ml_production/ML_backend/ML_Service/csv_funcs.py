import pandas as pd
from io import StringIO  # Import StringIO
from fastapi import UploadFile # For typing
from collections import Counter
from fastapi import UploadFile


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

		if ('Unnamed: 0' in df.columns):
			df = df.drop(columns=['Unnamed: 0']) # Weird column if you do some conversions
	except Exception as e:
		raise Exception(f"Could not read and convert csv file to pandas df. Error: {e}")
	return df

def validate_csv(df: pd.DataFrame) -> bool:
	data_col_schema = ['Type', 'Label','Headline','Byline','Section','Tagging','Paths','Publish Date','Body']
	try:
		current_df_col = list(df.columns)
		if (not Counter(current_df_col) == Counter(data_col_schema)):
			raise Exception(f"CSV Columns do not match the required data schema!")

		print("CSV looks good!")
		# [TODO]: Here we check for data duplication]
	except Exception as e:
		raise Exception(f"CSV Validation failed! {e}")
	return True

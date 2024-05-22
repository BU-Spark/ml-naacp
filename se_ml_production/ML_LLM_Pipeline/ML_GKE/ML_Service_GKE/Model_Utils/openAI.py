from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

class OpenAIClient:
    """
    A wrapper to manage the OpenAI API.
    """
    def __init__(self):
        self.client = None

    def createOpenAIClient(self, API_KEY=None):
        try:
            if (API_KEY == None):
                raise Exception("No API Key Given!")
            self.client = OpenAI(api_key=API_KEY,)
        except Exception as e:
            print(f"Failed to create OpenAI Client! {e}")
            raise Exception("Fatal Error in creating OpenAI Client.")
        return

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
    def get_embedding(self, text: str, model="text-embedding-3-small"):
        """
        Get the Ada Embedding.

        Note:
        Retry up to 10 times with exponential backoff, starting at 1 second and maxing out at 20 seconds delay
        """
        try:
            if (self.client == None):
                print(f"No OpenAI Client Specified! Returning!")
                return
            embedding = self.client.embeddings.create(input=text, model=model).data[0].embedding
            return embedding
        except Exception as e:
            print(f"Failed to retrieve ADA Embedding: {e}. Replacing with replacement value!")
            return [-1.0]
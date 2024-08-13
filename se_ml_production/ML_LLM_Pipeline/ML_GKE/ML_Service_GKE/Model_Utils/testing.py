from model_Loaders import load_llama_3_1


title = "The United States is a country in North America."

body = "The United States is a country primarily located in North America. It consists of 50 states, a federal district, five major unincorporated territories, 326 Indian reservations, and some minor possessions. At 3.8 million square miles (9.8 million square kilometers), it is the world's third- or fourth-largest country by total area. The United States shares significant land borders with Canada to the north and Mexico to the south as well as limited maritime borders with the Bahamas, Cuba, and Russia. With a population of more than 331 million people, it is the third most populous country in the world. The national capital is Washington, D.C., and the most populous city is New York City."

print("Loading Llama 3.1 8B model...")
# Load Llama 3.1 8B
llm = load_llama_3_1()

print("Running Llama 3.1 8B model...")
llama_prediction = llm.invoke({"headline": title, "body": body})

print(llama_prediction)

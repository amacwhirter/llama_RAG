# This code has two functions:
# 1. queryllm() - we demonstrate an example of setting detailed parameters of the LLM being used
# 2. chatcompl() - It generates a response to a given prompt using the OpenAI API using the llm.complete() method

from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file (optional)
load_dotenv()

# Access OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is retrieved
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable")

openai = OpenAI(api_key)

# Define your query
def queryllm():
    # Define your query
    query = "How was Microsoft founded?"

    llm = OpenAI(model="gpt-3.5-turbo",   # LLM model
                 query=query,                # Query text
                 temperature=0.5,            # Temperature parameter controls the randomness of the output
                 max_tokens=500,             # Maximum number of tokens to generate in the response text 
                 n=3,                        # Number of completions to generate for each prompt
                 stop=None                   # Stop parameter specifies a stop sequence for the generated text, e.g., stop="\n" to stop at the first newline 
                 ) 

    # Access the response text
    response = llm.complete(query)
    #completion = llm.choices[0].text.strip()   # Access the response text, strip() removes leading/trailing whitespaces

    # Print the response
    print(response)

def chatcompl():
    # Define your query
    response = OpenAI().complete("Roger Federer is ")
    print(response)

if __name__ == '__main__':
    #chatcompl()
    queryllm()

import os
from dotenv import load_dotenv

load_dotenv()
from llama_index.llms.openai import OpenAI

response = OpenAI(model="gpt-3.5-turbo",
                  openai_api_key=os.getenv("OPENAI_API_KEY")).complete("Who is Mahatma Gandhi?")
print(response)

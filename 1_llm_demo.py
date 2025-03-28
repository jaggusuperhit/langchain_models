import os
from dotenv import load_dotenv
from langchain_openai import OpenAI

load_dotenv()  # Load environment variables from .env file
llm = OpenAI(model="gpt-3.5-turbo")
result = llm.invoke("What is the capital of India?")
print(result)

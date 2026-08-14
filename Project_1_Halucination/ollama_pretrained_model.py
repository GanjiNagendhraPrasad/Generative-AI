'''
In this file we are going to take the Open AI model using API Key
and will use the model as it is and also will make the Model as Halucinate
'''
import numpy as np
import langchain
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
os.environ["openai_API_key"] = os.getenv("openai_API_key")
from langchain_core.output_parsers import StrOutputParser

ollama_model = ChatOllama(model = "gemma:2b")
# in openai_model we have pretrained model

Task = """I want to Develop a AI Company but I dont have any education 
                backgroup and even no idea How to maintain the 
                             Administration to handle the Employess"""
# result1 = openai_model.invoke(Task)
# print(result1.content)

print("=======================================================================")

temp = ChatPromptTemplate(
    [
        ("system" , "Behave that you have 30 years Experience in running AI Company with maintaning of 500 Employees"),
        ("user" , "{input}")
    ]
)

output = StrOutputParser()

reg = temp | ollama_model | output

result = reg.invoke({"input" : Task})
print(result)
'''
In this file we are going to take the Open AI model using API Key
and will use the model as it is and also will make the Model as Halucinate
'''
import numpy as np
import langchain
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ["openai_API_key"] = os.getenv("openai_API_key")

openai_model = ChatOpenAI(model = "gpt-4o") # in openai_model we have pretrained model
#result=openai_model.invoke('i want to learn MBA in 2026')
#result=openai_model.invoke('''I want to Develop a AI Company but I dont have any education
                #backgroup and even no idea How to maintain the
                             #Administration to handle the Employess''')
#print(result.content)

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

output = StrOutputParser()# this removes the unwanted content. instead of writing {.content} use this

reg = temp | openai_model | output

result = reg.invoke({"input" : Task})
print(result)










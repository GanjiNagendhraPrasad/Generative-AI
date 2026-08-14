'''
In this file we are going to give our data to open ai embedding models and it
will convert into vectors
'''
import numpy as np
import langchain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import dotenv
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["openai_API_key"] = os.getenv("openai_API_key")

# s = TextLoader("./../data_location/text_sample.txt")
# openai_embed_model = OpenAIEmbeddings(model = "text-embedding-3-small")
# t = ''
# for i in s.load():
#     t = t + i.page_content
# result = openai_embed_model.embed_query(t)
# print(result)
# print(len(result)) # 1536 vectors

# s = 'I love my country and want to stay here for 20 years'
#openai_embed_model = OpenAIEmbeddings(model = "text-embedding-3-small")
openai_embed_model = OpenAIEmbeddings(model = "text-embedding-3-small" , dimensions = 300)#instead of 1536,{300} taken.
# result = openai_embed_model.embed_query(s)
# print(result)
# print(len(result))

# s = PyPDFLoader("./../data_location/Complete_74_AI_Video_Prompts_Collection.pdf")
# t = []
# for i in s.load():
#     t.append(openai_embed_model.embed_query(i.page_content))
# # t = [  [1536] , [1536] , [1536] , [1536] , [1536] ]
# print(f"Total Pages in the PDF : {len(t)}")
# print(len(t[0]))
# print(len(t[1]))

s = TextLoader("./../data_location/text_sample.txt")
chunk_obj = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=50)
result = chunk_obj.split_documents(s.load())
print(f"Number of Chunks : {len(result)}")

t = []
for i in result:
    t.append(openai_embed_model.embed_query(i.page_content))
print(len(t))  # t = [ [300] , [] , [] ................. []]
print(len(t[0]))
print(len(t[1]))


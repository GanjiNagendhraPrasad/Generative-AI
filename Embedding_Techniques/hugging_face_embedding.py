'''
In this file we are going to give our data to Hugging Face embedding models and it
will convert into vectors
'''
import numpy as np
import langchain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import dotenv
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["openai_API_key"] = os.getenv("openai_API_key")
os.environ["hugging_face_API"] = os.getenv("hugging_face_API")

s = TextLoader("./../data_location/text_sample.txt")

hugg_embed_model = HuggingFaceEmbeddings(model = "all-MiniLM-L6-v2")
t = [] # t = [0.01 , 0.0.2 ..........]
# t = [   [0.01,0.02]   , [0.0 , 0.2]    ]
for i in s.load():
    t.append(hugg_embed_model.embed_query(i.page_content))
print(t)
print(len(t[0]))


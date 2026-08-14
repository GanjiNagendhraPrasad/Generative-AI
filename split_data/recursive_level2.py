'''
Using Recursive text splitter we are going to convert the data into chunks
what is Chunk : Chunk means dividing the data into seperate pieces
'''
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import langchain
import langchain_community
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

s = "I love My country and want to stay in india for next 100 years"

chunk_obj = RecursiveCharacterTextSplitter(chunk_size=10 , chunk_overlap=5)
#result = chunk_obj.create_documents([s])
result=chunk_obj.split_text(s)

print(f"Number of Chunks [documents] : {len(result)}")
print(result)
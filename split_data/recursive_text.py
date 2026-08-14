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

s=TextLoader('./../data_location/text_sample.txt')
# s.load() only 1 document

chunk_obj = RecursiveCharacterTextSplitter(chunk_size=100 , chunk_overlap=50)
result = chunk_obj.split_documents(s.load())

print(f"Number of Chunks [documents] : {len(result)}")
print(result[0].page_content)
print(result[1].page_content)
print(result[2].page_content)
print(result[3].page_content)
print(result[4].page_content)
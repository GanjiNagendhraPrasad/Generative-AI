'''
Text file (data) -> 1doc -> chunks -> (multiple docs) -> vectors -> chromadb
'''
import numpy as np
import langchain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma, FAISS

s=TextLoader('./../data_location/text_sample.txt')
# in s we have only single document

chunk_obj=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=50)
multiple_docs=chunk_obj.split_documents((s.load()))
# [ [doc1] , [doc2] , [doc3] ......]
# print(multiple_docs)
# print(len(multiple_docs))  # 17 doocs

ollama_obj=OllamaEmbeddings(model="nomic-embed-text-v2-moe")
# vector_list=[]
# for i in multiple_docs:
#     vector_list.append(ollama_obj.embed_query(i.page_content))
# print(len(vector_list)) #17 [ [..768] , [..768] , .....]
# print(len(vector_list[0]))

faiss_db =  FAISS.from_documents(multiple_docs,ollama_obj)

sample_question='what is python'

#result=chr_db.similarity_search(sample_question,k=3)
result=faiss_db.similarity_search_with_score(sample_question,k=3)
print(result)

# save faiss db in local
faiss_db.save_local('FAISS_DATABASE_LOCALLY')














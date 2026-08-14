'''
In this file, we are going to take the text data (either directly text or the documents)
and give it to the OpenAI embedded model, and it will convert it into vectors.
OpenAI is completely an open-source,
free library where we can get the free LLMs and also free embedded models.
'''
import numpy as np
import langchain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader

#s = TextLoader("./../data_location/text_sample.txt")
# olama_obj = OllamaEmbeddings(model = "nomic-embed-text-v2-moe")
# vectors_list = [] # [  [....]  , [.....] , [....] ] => 124
# for i in s.load():
#     vectors_list.append(olama_obj.embed_query(i.page_content))
# print(len(vectors_list))
# print(vectors_list)
# print(len(vectors_list[0]))

s = PyPDFLoader("./../data_location/Complete_74_AI_Video_Prompts_Collection.pdf")
total_text_data = ''
for i in s.load():
    total_text_data = total_text_data + i.page_content + '\n'
chunk_obj = RecursiveCharacterTextSplitter(chunk_size=100 , chunk_overlap=50)
chunk_results = chunk_obj.create_documents([total_text_data])
# print(chunk_results) # => 124 =>  [  [chunk_obj] , [chunk_obj] , [..] , [....] ]
# print(len(chunk_results))
# print(chunk_results[0])
# print(chunk_results[1])

olama_obj = OllamaEmbeddings(model = "nomic-embed-text-v2-moe")
vectors_list = [] # [  [....]  , [.....] , [....] ] => 124
for i in chunk_results:
    vectors_list.append(olama_obj.embed_query(i.page_content))
print(len(vectors_list))
print(len(vectors_list[0]))
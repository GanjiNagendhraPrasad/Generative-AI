'''
In this file, we are going to save the converted vectors into the chroma db.
'''

import numpy as np
import langchain
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

s = TextLoader("./../data_location/text_sample.txt") # 1 doc
vector_list = []  # [  [........768]  ]

ollama_obj = OllamaEmbeddings(model = "nomic-embed-text-v2-moe")
for i in s.load():
    vector_list.append(ollama_obj.embed_query(i.page_content))

chroma_db = Chroma.from_documents(s.load() , ollama_obj) # 768 vectors will save in chroma db
print(chroma_db)

sample_input = "what is python"
result = chroma_db.similarity_search(sample_input)
print(result)
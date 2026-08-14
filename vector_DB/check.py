import numpy as np
import langchain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma,FAISS

ollama_obj=OllamaEmbeddings(model="nomic-embed-text-v2-moe")

#load_chroma_database=Chroma(persist_directory='./chroma_database', embedding_function=ollama_obj)

loaded_faiss_db=FAISS.load_local('FAISS_DATABASE_LOCALLY',ollama_obj,allow_dangerous_deserialization=True)


sample_question='what is python'
result=loaded_faiss_db.similarity_search_with_score(sample_question,k=3)
print(result)
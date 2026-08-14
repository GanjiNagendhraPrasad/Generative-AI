'''
In this file we are going to take custom data via RAG and give to LLM
'''
import numpy as np
import langchain
from langchain_community.document_loaders import TextLoader,PyPDFLoader,CSVLoader,WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["openai_API_key"] = os.getenv("openai_API_key")

# Stage 1 in RAG -> Loading the data
data_from_internet = WebBaseLoader("https://www.tpointtech.com/python-tutorial")
#print(len(data_from_internet.load()))
# so the entire data from the internet API is in the form of 1 document
# for i in data_from_internet.load():
#     print(i.page_content)

# Stage 2 in RAG -> Chunks
chunk_obj = RecursiveCharacterTextSplitter(chunk_size=300 , chunk_overlap=100)
individual_chunks = chunk_obj.split_documents(data_from_internet.load())
#print(len(individual_chunks))
# Know 1 document is converted into 106 document each document we have Page content

# Stage 3 in RAG -> 106 Documents we are going to convert into 106 Embedded vectors
openai_embed_model = OpenAIEmbeddings(model = "text-embedding-3-small")
all_vectors = []
# before : all_vectors = []
for i in individual_chunks:
    all_vectors.append(openai_embed_model.embed_query(i.page_content))
# after all_vectors = [  [.,.,.,....1536]  , [.,.,.,....1536] , [] , [] , .............. []]
# print(all_vectors)
# print(len(all_vectors[0]))
# print(len(all_vectors[1]))

# Stage 4: Save the Vectos in the DB (ChromaDB)
chroma_db = Chroma.from_documents(individual_chunks , openai_embed_model)
sample_input = "Why Python is very good programming language"
result = chroma_db.similarity_search(sample_input,k = 3)
# print(result)
# print('=============================')
# print(result[0])
# print(result[1])
# print(result[2])

# loading pretrained OpenAI LLM
openai_llm = ChatOpenAI(model = "gpt-4o")

prompt = ChatPromptTemplate.from_template(
    """
      Answer the queries from the below content:
      <context>
      {context}
      </context>

      Question: {input}
      """
)

document_chain = create_stuff_documents_chain(openai_llm,prompt)

response = document_chain.invoke({
    "input":sample_input,
    "context":result
}
)
print(response)












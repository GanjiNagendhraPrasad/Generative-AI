'''
In this file we are going to load the data from PDF file using langchain
'''
# In Pdf file each page will be converted as 1 document
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import langchain
import langchain_community
from langchain_community.document_loaders import PyPDFLoader

s=PyPDFLoader('./../data_location/Complete_74_AI_Video_Prompts_Collection.pdf')
#print(s)
#print(s.load())
print(f"Total Pages in PDF file was : {len(s.load())}")

with open("Saving_pdf_data_to_text_file.txt","a") as f:
    t = 1
    for i in s.load():
        print(i.page_content)
        f.write("Data From Page : "+ str(t) + " = "+ i.page_content + "\n")
        t = t + 1
        print("==========================================")
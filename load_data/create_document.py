'''
In this file we are going to create document from own data
'''
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import langchain
import langchain_community
from langchain_core.documents import Document

# data = "I love My Country which is India"
# sol = Document(page_content=data , metadata={"source": "https://example.com"})
# print(sol)
# print(type(sol))
# # above created our document from string
# for i in sol:
#     print(i)

with open("./../data_location/text_sample.txt","r") as f:
    m = f.read()

# in variable m we have text data

sol = Document(page_content=m , metadata={"source": "https://example.com"})
print(sol)
# in sol the text data is converted into document

string_only = sol.page_content
print(string_only)

# String_only -> from document we are pulling only text

with open("Check.txt","w") as f1:
    f1.write(string_only)
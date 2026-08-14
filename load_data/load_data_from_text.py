'''
In this file we are going to load the data from text file using
langchain Framework
'''
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import langchain
import langchain_community
from langchain_community.document_loaders import TextLoader

sol=TextLoader('./../data_location/text_sample.txt')
# print(sol)
# print(sol.load())
r=sol.load()
for i in r:
    print(i.page_content)

'''
If we have text file langchain will read the data from the text file 
in 1 Document format 
'''
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import langchain
import langchain_community
from langchain_community.document_loaders import CSVLoader

s=CSVLoader('./../data_location/Titanic-Dataset.csv')

# 891 rows and 12 columns
print(f"Number of Documents in the data : {len(s.load())}")
print(f"Data from 1 document")
print(s.load()[0].page_content)

# for i in s.load():
#     print(i.page_content)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 20:02:21 2024

@author: sn22wex
"""

import os 
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from secret_key import openapi_key


os.environ['OPENAI_API_KEY'] = openapi_key

llm = OpenAI(temperature=0.9, max_tokens=500)

loaders = UnstructuredURLLoader(urls = [
    "https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html",
    "https://www.moneycontrol.com/news/business/tata-motors-mahindra-gain-certificates-for-production-linked-payouts-11281691.html"
    ])

data = loaders.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200)

docs = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings()

vectorindex_openai =  FAISS.from_documents(docs, embeddings)

# file_path = "vector_index.pkl"

# with open(file_path, "wb") as f:
#     pickle.dump(vectorindex_openai, f)
    

vectorindex_openai.save_local("faiss_store")

vectorindex = FAISS.load_local("faiss_store", embeddings, allow_dangerous_deserialization=True)

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever = vectorindex.as_retriever())
 
query = "How many sectors does production-linked incentive programme cover?"

langchain.debug = True
 
result = chain({"question" : query}, return_only_outputs=True)

print(result)
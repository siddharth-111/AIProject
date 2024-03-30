#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:07:27 2024

@author: sn22wex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:55:15 2024

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

from dotenv import load_dotenv

load_dotenv()


st.title("new research tool")
st.sidebar.title("News article URLS")

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)
    
process_url_clicked =  st.sidebar.button("Process URLs")
file_path = "faiss_user_tool_store"
file_path_index = "faiss_user_tool_store/index.pkl"

main_placefolder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)


if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data loading... Started")
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
       separators=["\n\n", "\n", ".", ","],
       chunk_size=1000)
    
    
    main_placefolder.text("Text Splitter... Started")
    docs = text_splitter.split_documents(data)
    
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placefolder.text("Embedding vector.. Started")
    time.sleep(2)
    
    vectorstore_openai.save_local(file_path)
    
query = main_placefolder.text_input("Question: ")

if query:
    if os.path.exists(file_path_index):
        
        embeddings = OpenAIEmbeddings() 
        vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
        
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever = vectorstore.as_retriever())
        
        langchain.debug = True
        
        result = chain({"question" : query}, return_only_outputs=True)
        
        st.header("Answer")
        st.subheader(result["answer"])
        
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources: ")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
    
     

    
       
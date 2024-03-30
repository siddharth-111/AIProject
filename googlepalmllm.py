#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:32:01 2024

@author: sn22wex
"""
import logging

from langchain import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

llm = OpenAI(temperature=0.9, max_tokens=500)
# poem = llm("Write a 5 line poem for my love of samosa")
# print(poem)


logging.basicConfig(level=logging.DEBUG)

embeddings = HuggingFaceInstructEmbeddings()

vectordb_file_path="faiss_csv_llm_index"

def create_vector_db():
    loader = CSVLoader(file_path="encoded-codebasics_faqs.csv", source_column='prompt')
    data = loader.load()
    vectordb = FAISS.from_documents(data, embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain(): 
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)

    retriever = vectordb.as_retriever()

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    ) 

    chain= RetrievalQA.from_chain_type(llm=llm, chain_type = "stuff", retriever=retriever, input_key="query", return_source_documents=True,
                                   chain_type_kwargs={"prompt": PROMPT})
    
    return chain

if __name__ == "__main__":
    chain = get_qa_chain()
    
    response = chain("do you provide internship? Do you have EMI option")
    print(response)
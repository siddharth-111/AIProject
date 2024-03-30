import logging

from langchain import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceEndpoint

load_dotenv()

# llm = OpenAI(temperature=0.9)
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id, temperature=0.1, token='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'
)

def get_tax_saving_chain(): 
    # loader = PyPDFLoader("tax-document-germany-2023-08-17.pdf")
    loader = PyPDFLoader("lumpsum.pdf")
    pages = loader.load()
    embeddings = HuggingFaceInstructEmbeddings()
    # embeddings = OpenAIEmbeddings()
    vectordb_file_path="faiss_tax_llm_index"

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, 
    chunk_overlap=200)

    docs = text_splitter.split_documents(pages)

    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(vectordb_file_path)

    vectordb_output = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb_output.as_retriever()

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, input_key="query", return_source_documents=True)

    return chain

if __name__ == "__main__":
    chain = get_tax_saving_chain()
    
    response = chain("How many working days can i claim for home office lump sum")
    print(response)
import streamlit as st
from googlepalmllm import create_vector_db, get_qa_chain


st.title("QA section")
btn = st.button("Create knowledgebase")
if btn:
    pass

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer: ")
    st.write(response["result"])
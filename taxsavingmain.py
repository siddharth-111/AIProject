import streamlit as st
import langchain
from taxsavingtest import get_tax_saving_chain

st.title("QA section")
btn = st.button("Create knowledgebase")
if btn:
    pass

question = st.text_input("Question: ")

if question:
    chain = get_tax_saving_chain()
    langchain.debug = True
    response = chain(question)

    st.header("Answer: ")
    st.write(response["result"])
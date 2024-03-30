#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:50:26 2024

@author: sn22wex
"""

import os
from secret_key import openapi_key


os.environ['OPENAI_API_KEY'] = openapi_key

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain

llm = OpenAI(temperature=0.6)

# name = llm("I want to open restaurant for Indian food. Suggest a fancy name for this.")
# print(name)

def generate_restaurant_name_and_items(cuisine):
    

    prompt_template_name = PromptTemplate(
        input_variables = ['cuisine'],
        template = " want to open restaurant for {cuisine} food. Suggest a fancy name for this."
        )
    
    
    
    name_chain = LLMChain(llm = llm, prompt=prompt_template_name, output_key="restaurant_name")
    
    prompt_template_items = PromptTemplate(
        input_variables = ['restaurant_name'],
        template = "Suggest some menu items for {restaurant_name}. Return it as a comma separated value"
        )
    
    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")
    
    # chain = SimpleSequentialChain(chains=[name_chain, food_items_chain])
    
    chain = SequentialChain( 
        chains = [name_chain, food_items_chain],
        input_variables = ['cuisine'],
        output_variables = ['restaurant_name', 'menu_items'])
    
    response = chain({'cuisine': cuisine})
    
    return response

if __name__ == "__main__": 
    print(generate_restaurant_name_and_items("Italian"))
    
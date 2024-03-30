#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 19:28:14 2024

@author: sn22wex
"""

import os
from secret_key import openapi_key
from secret_key import serpapi_key

os.environ['OPENAI_API_KEY'] = openapi_key
os.environ['SERPAPI_API_KEY'] = serpapi_key

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.6)

tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(
    tools, 
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# my_respone = agent.run("When was elon musk born and what is his age in 2024")
# print(agent.run("When was elon musk born and what is his age in 2024"))
agent.run("What was the GDP of US in 2022")

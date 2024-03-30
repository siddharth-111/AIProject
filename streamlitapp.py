#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 15:38:46 2024

@author: sn22wex
"""

import streamlit as st
import langchainhelper


st.title("Restaurant name generator")

cuisine = st.sidebar.selectbox("Pick a cuisine", ("Indian", "American", "Arabic", "Chinese", "Mexican"))


def generate_restaurant_name_and_items(cuisine) : 
    return langchainhelper.generate_restaurant_name_and_items(cuisine)


if cuisine:
    response = generate_restaurant_name_and_items(cuisine)
    st.header(response['restaurant_name'].strip())
    menu_items = response['menu_items'].split(",")
    st.write("menu_items")
    for item in menu_items:
        st.write("-", item)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 21:26:55 2024

@author: sn22wex
"""


from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex


import pandas as pd
import numpy as np


# model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)

loader = TextLoader("nvda_news_1.txt")
data = loader.load()

# UnstructuredURLLoader(urls = [])
text = "Interstellar is a 2014 epic science fiction film co-written, directed, and co-produced by Christopher Nolan. It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, Michael Caine, and Matt Damon. Set in a dystopian future where humanity is embroiled in a catastrophic blight and famine, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for humankind. Brothers Christopher and Jonathan Nolan wrote the screenplay, which had its origins in a script Jonathan developed in 2007 and was originally set to be directed by Steven Spielberg. Kip Thorne, a Caltech theoretical physicist and 2017 Nobel laureate in Physics,[4] was an executive producer, acted as a scientific consultant, and wrote a tie-in book, The Science of Interstellar. Cinematographer Hoyte van Hoytema shot it on 35 mm movie film in the Panavision anamorphic format and IMAX 70 mm. Principal photography began in late 2013 and took place in Alberta, Iceland, and Los Angeles. Interstellar uses extensive practical and miniature effects, and the company Double Negative created additional digital effects. Interstellar premiered in Los Angeles on October 26, 2014. In the United States, it was first released on film stock, expanding to venues using digital projectors. The film received positive reviews from critics and grossed over $681 million worldwide ($703 million after subsequent re-releases), making it the tenth-highest-grossing film of 2014. It has been praised by astronomers for its scientific accuracy and portrayal of theoretical astrophysics.[5][6][7] Interstellar was nominated for five awards at the 87th Academy Awards, winning Best Visual Effects, and received numerous other accolades."

# splitter = CharacterTextSplitter(
#     separator=".",
#     chunk_size=200,
#     chunk_overlap=0
# )

# chunks = splitter.split_text(text)
# print(len(chunks))


r_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "],
                                            chunk_size=200,
                                            chunk_overlap=0
                                            )

chunks = r_splitter.split_text(text)
# print(len(chunks))

pd.set_option('display.max_colwidth', 100)

df = pd.read_csv("sample_text.csv")


encoder = SentenceTransformer("all-mpnet-base-v2")

vectors = encoder.encode(df.text)

dim = vectors.shape[1]

# index = faiss.IndexFlatL2(dim)

# index.add(vectors)

index = AnnoyIndex(dim, 'euclidean')

for i, vector in enumerate(vectors):
    index.add_item(i, vector)
    
index.build(10)

search_query = "An apple a day keeps a doctor away"

vec = encoder.encode(search_query)

nearest_ids = index.get_nns_by_vector(vec, 2)
nearest_ids, distances = index.get_nns_by_vector(vec, n=2, include_distances=True)

# svec = np.array(vec).reshape(1, -1)
 
print(df.loc[nearest_ids])


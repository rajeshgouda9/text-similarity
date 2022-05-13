import streamlit as st
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.max_seq_length = 512

st.title('Text Similarity Checker')
title1 = st.text_area('Enter First Text')
title2 = st.text_area('Enter Second Text')
if st.button('Check'):
    embedding1 = model.encode(title1, convert_to_tensor=True)
    embedding2 = model.encode(title2, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embedding1, embedding2)
    a = cosine_scores.item()
    st.header("{:.2f}".format(a))
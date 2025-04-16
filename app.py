import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
import os

# Streamlit app
st.title("Cultureach:")
st.write("Enter a sentence to see value labels from two different approaches.")

# Cache model to load once
@st.cache_resource
def load_model():
    return SentenceTransformer('BAAI/bge-base-en-v1.5', device='cpu')

model = load_model()

# Load precomputed embeddings and value titles
@st.cache_data
def load_embeddings_and_titles():
    # Load embeddings
    embeddings_with_keywords_np = np.load('value_embeddings_with_keywords.npy')
    embeddings_no_keywords_np = np.load('value_embeddings_no_keywords.npy')
    
    # Load value titles (fallback if values.csv missing)
    if os.path.exists('values.csv'):
        values_df = pd.read_csv('values.csv')
        value_titles = values_df['Title'].tolist()
    
    return embeddings_with_keywords_np, embeddings_no_keywords_np, value_titles

# Load data
embeddings_with_keywords_np, embeddings_no_keywords_np, value_titles = load_embeddings_and_titles()

# Initialize FAISS indexes
dimension = embeddings_with_keywords_np.shape[1]  # 768
index_with_keywords = faiss.IndexFlatL2(dimension)
index_no_keywords = faiss.IndexFlatL2(dimension)
index_with_keywords.add(embeddings_with_keywords_np)
index_no_keywords.add(embeddings_no_keywords_np)
# threshold = 0.7
# Input form
input_text = st.text_area("Enter a sentence:", "")
if st.button("Predict Values"):
    if input_text:
        # Embed input
        input_embedding = model.encode([input_text], convert_to_tensor=True, show_progress_bar=False)
        input_embedding_np = input_embedding.cpu().numpy()
        
        # Search both indexes (top-1)
        k = 3
        distances_with, indices_with = index_with_keywords.search(input_embedding_np, k)
        distances_no, indices_no = index_no_keywords.search(input_embedding_np, k)
        
        # Filter by threshold
        # labels_with = []
        # confidences_with = []
        # for dist, idx in zip(distances_with[0], indices_with[0]):
        #     if dist < threshold:
        #         labels_with.append(value_titles[idx])
        #         confidences_with.append(1 / (1 + dist))
        
        # labels_no = []
        # confidences_no = []
        # for dist, idx in zip(distances_no[0], indices_no[0]):
        #     print(dist)
        #     if dist < threshold:
        #         labels_no.append(value_titles[idx])
        #         confidences_no.append(1 / (1 + dist))
        labels_with = [value_titles[idx] for idx in indices_with[0]]
        # confidences_with = [max(0, 1 - min(dist, 2) / 2) * 100 for dist in distances_with[0]]
        distances_with_list = distances_with[0].tolist()
        
        labels_no = [value_titles[idx] for idx in indices_no[0]]
        # confidences_no = [(1 / (1 + dist)) * 100 for dist in distances_no[0]]
        distances_no_list = distances_no[0].tolist()
        # Display results side-by-side
        st.subheader("Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Approach 1**")
            if labels_with:
                for label, conf in zip(labels_with, distances_with_list):
                    st.success(f"**Value**: {label}")
                    st.write(f"**Distance**: {conf:.2}")
            else:
                st.warning("No values below threshold.")
        
        with col2:
            st.write("**Approach 2**")
            if labels_no:
                for label, conf in zip(labels_no, distances_no_list):
                    st.success(f"**Value**: {label}")
                    st.write(f"**Distance**: {conf:.2}")
            else:
                st.warning("No values below threshold.")
    else:
        st.error("Please enter a sentence.")

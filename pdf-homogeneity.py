import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
import PyPDF2

abstracts = []
uploaded_file = st.file_uploader("Choose a file", type=['pdf'])
if uploaded_file is not None:
    # Read PDF content and extract text
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    abstract = ' '.join(page.extract_text() for page in pdf_reader.pages)
    abstracts.append(abstract)

# Convert abstracts into a list of strings
abstracts_text = abstracts

# Create a TF-IDF vectorizer to convert text data into numerical features
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(abstracts_text)

# Perform KMeans clustering if you have enough samples
num_clusters = 1  # You can adjust the number of clusters
if X.shape[0] >= num_clusters:
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels_pred = kmeans.fit_predict(X)
    labels_true = labels_pred

    # Evaluate the homogeneity of the clusters
    homogeneity = homogeneity_score(labels_true, labels_pred)
    completeness = completeness_score(labels_true, labels_pred)
    measure = v_measure_score(labels_true, labels_pred)
    st.write(f"Homogeneity Score: {homogeneity}")
    st.write(f"Completeness Score: {completeness}")
    st.write(f"Measure Score: {measure}")
else:
    print("Not enough samples to perform clustering with the specified number of clusters.")

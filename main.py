import cv2
import base64
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


def decode_array(s: str):
    b = base64.b64decode(s.encode('utf8'))
    return np.frombuffer(b, dtype=np.float64)


def img_path_to_bovw(image: np.ndarray, kmeans: KMeans, sift: cv2.SIFT) -> np.ndarray:
    # Get SIFT descriptors
    _, descriptors = sift.detectAndCompute(image, None)
    predictions = kmeans.predict(descriptors)

    # Compute histogram
    histogram = np.bincount(predictions, minlength=kmeans.n_clusters)
    histogram = histogram / np.sum(histogram)
    return histogram


def load_rbg(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


# Load Kmeans from disk
kmeans: KMeans = joblib.load('kmeans_1000_512.joblib')
# Create SIFT object
sift = cv2.SIFT.create()

# Load database from disk
df = pd.read_csv('db.csv', index_col=0)
df['vector'] = df['vector'].map(decode_array)

# Initialize Nearest Neighbors
neighbors = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
neighbors.fit(np.vstack(df['vector'].values), df.index.values)

st.title('Nyandex Images')
file = st.file_uploader('Drop an image here')

if file is not None:
    # Read buffer
    buf = np.frombuffer(file.getbuffer(), dtype=np.uint8)
    # Decode image from buffer
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    st.image(img)

    st.title('10 most similar images')
    # Compute visual bovw for image
    target = img_path_to_bovw(img, kmeans, sift)
    # Find 10 most similar images
    _, indices = neighbors.kneighbors([target], n_neighbors=10)
    # Display results
    for idx in indices.flat:
        st.image(load_rbg(df.loc[idx]['path']))

import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Function to scrape book data
def scrape_book_data(base_url="http://books.toscrape.com"):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    books = []
    for product in soup.select('article.product_pod'):
        img_element = product.select_one('img.thumbnail')
        title = img_element['alt']
        img_url = base_url + '/' + img_element['src'].replace('..', '')
        description = 'Description not available'
        books.append({'title': title, 'img_url': img_url})
    return books

# Function to download images
def download_images(image_urls):
    images = []
    for url in image_urls:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        images.append(img)
    return images

# Function to extract features from images
def extract_features(images):
    model = VGG16(weights='imagenet', include_top=False, pooling='max')
    features = []
    for img in images:
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        feature = model.predict(img_array)
        features.append(feature.flatten())
    return np.array(features)

# Function to cluster images
def cluster_images(features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels

# Function to extract keywords for each cluster
def extract_keywords_for_cluster(descriptions):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(descriptions)
    keywords = vectorizer.get_feature_names_out()
    return keywords

# Main Streamlit UI code
def main():
    st.title("Book Image Clusters")

    # Scrape book data, download images, and extract features
    st.write("Scraping book data and extracting features...")
    books = scrape_book_data()
    images = download_images([book['img_url'] for book in books])
    features = extract_features(images)

    # Cluster images and group books by cluster labels
    st.write("Clustering images and grouping books...")
    labels = cluster_images(features)
    clusters = {}
    for book, label in zip(books, labels):
        if label not in clusters:
            clusters[label] = {'books': [], 'descriptions': []}
        clusters[label]['books'].append(book)
        clusters[label]['descriptions'].append(book['title'])

    # Extract keywords for each cluster
    for cluster_id, cluster_data in clusters.items():
        descriptions = cluster_data['descriptions']
        cluster_data['keywords'] = extract_keywords_for_cluster(descriptions) if descriptions else []

    # Display clusters
    st.write("Displaying clusters...")
    for cluster_id, cluster_data in clusters.items():
        st.subheader(f"Cluster {cluster_id}")
        keywords = ", ".join(cluster_data['keywords'])
        st.write(f"Keywords: {keywords}")
        st.image([book['img_url'] for book in cluster_data['books']], width=200)

if __name__ == '__main__':
    main()

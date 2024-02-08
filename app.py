from flask import Flask, render_template_string
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
from io import BytesIO
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array


# Initialize Flask app
app = Flask(__name__)


# Define the HTML template for rendering the clusters
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Book Image Clusters</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
        }
        .container {
            width: 80%;
            margin: auto;
            overflow: hidden;
        }
        h1 {
            text-align: center;
            padding: 20px;
        }
        .cluster {
            margin-bottom: 30px;
            overflow: hidden;
        }
        .cluster h2 {
            text-align: center;
            background-color: #007bff;
            color: white;
            padding: 10px 0;
            margin: 0 0 15px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.4);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            grid-gap: 10px;
            align-items: stretch;
        }
        .grid img {
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }
        .grid img:hover {
            transform: scale(1.05);
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Book Image Clusters</h1>
        {% for cluster_id, image_urls in clusters.items() %}
            <div class="cluster">
                <h2>Cluster {{ cluster_id }}</h2>
                <div class="grid">
                    {% for img_url in image_urls %}
                        <img src="{{ img_url }}" alt="Book Image">
                    {% endfor %}
                </div>
            </div>
        {% endfor %}
    </div>
</body>
</html>
"""


def scrape_book_images(base_url="http://books.toscrape.com"):
    """Scrape book images from the given URL."""
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    image_urls = [base_url + "/" + img["src"] for img in soup.select("img.thumbnail")]
    return image_urls


def download_images(image_urls):
    """Download images from a list of URLs."""
    images = []
    for url in image_urls:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        images.append(img)
    return images


def extract_features(images):
    """Extract features from images using VGG16."""
    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='max')
    features = []
    for img in images:
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features.append(model.predict(img_array)[0])
    return np.array(features)


def cluster_images(features, n_clusters=5):
    """Cluster images based on extracted features."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels


@app.route('/')
def index():
    image_urls = scrape_book_images()
    images = download_images(image_urls)
    features = extract_features(images)
    labels = cluster_images(features)


    clusters = {i: [] for i in range(max(labels) + 1)}
    for url, label in zip(image_urls, labels):
        clusters[label].append(url)


    return render_template_string(HTML_TEMPLATE, clusters=clusters)


if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, render_template_string
# import requests
# from bs4 import BeautifulSoup
# from sklearn.cluster import KMeans
# from sklearn.feature_extraction.text import CountVectorizer
# import numpy as np
# from PIL import Image
# from io import BytesIO
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array

# app = Flask(__name__)

# # Define the HTML template for rendering the clusters
# # Define the HTML template for rendering the clusters
# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html>
# <head>
#     <title>Book Image Clusters</title>
#     <link href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" rel="stylesheet">
#     <style>
#         body {
#             background-color: #f8f9fa;
#         }
#         .container {
#             padding-top: 2rem;
#         }
#         .cluster {
#             margin-bottom: 1rem;
#             background-color: white;
#             border-radius: 0.25rem;
#             box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
#             padding: 1rem;
#         }
#         .cluster-header {
#             margin-bottom: 1rem;
#             border-bottom: 1px solid #eee;
#             padding-bottom: 0.5rem;
#         }
#         .cluster h2 {
#             font-size: 1.5rem;
#             color: #007bff;
#             margin-bottom: 0;
#         }
#         .keywords {
#             font-style: italic;
#             color: #6c757d;
#         }
#         .book {
#             text-align: center;
#             padding: 0.5rem;
#         }
#         .book img {
#             max-width: 100%;
#             height: auto;
#             margin-bottom: 0.5rem;
#             border-radius: 0.25rem;
#             box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
#         }
#         .book-title {
#             font-size: 1rem;
#             font-weight: bold;
#             color: #333;
#         }
#         .description {
#             font-size: 0.875rem;
#             color: #6c757d;
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <h1 class="text-center mb-4">Book Image Clusters</h1>
#         {% for cluster_id, books in clusters.items() %}
#             <div class="cluster">
#                 <div class="cluster-header">
#                     <h2>Cluster {{ cluster_id }}</h2>
#                     <p class="keywords">Keywords: {{ keywords|join(", ") }}</p>
#                 </div>
#                 <div class="row">
#                     {% for book in books %}
#                         <div class="col-md-3 book">
#                             <img src="{{ book['img_url'] }}" alt="Book Image">
#                             <p class="book-title">{{ book['title'] }}</p>
#                             <p class="description">{{ book['description'] }}</p>
#                         </div>
#                     {% endfor %}
#                 </div>
#             </div>
#         {% endfor %}
#     </div>
#     <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js"></script>
#     <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js"></script>
#     <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"></script>
# </body>
# </html>
# """

# # ... rest of your Flask application code ...


# def scrape_book_data(base_url="http://books.toscrape.com"):
#     """Scrape book titles, images, and descriptions from the given URL."""
#     response = requests.get(base_url)
#     soup = BeautifulSoup(response.content, 'html.parser')
#     books = []
#     for product in soup.select('article.product_pod'):
#         img_element = product.select_one('img.thumbnail')
#         title = img_element['alt']
#         img_url = base_url + '/' + img_element['src'].replace('..', '')
#         # Placeholder for description - this would require scraping individual book pages
#         description = 'Description not available'
#         books.append({'title': title, 'img_url': img_url})
#     return books

# def download_images(image_urls):
#     """Download images from a list of URLs."""
#     images = []
#     for url in image_urls:
#         response = requests.get(url)
#         img = Image.open(BytesIO(response.content)).convert('RGB')  # Convert to RGB
#         images.append(img)
#     return images

# def extract_features(images):
#     """Extract features from images using VGG16."""
#     model = VGG16(weights='imagenet', include_top=False, pooling='max')
#     features = []
#     for img in images:
#         img = img.resize((224, 224))
#         img_array = img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = preprocess_input(img_array)
#         feature = model.predict(img_array)
#         features.append(feature.flatten())
#     return np.array(features)

# def cluster_images(features, n_clusters=5):
#     """Cluster images based on extracted features."""
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     labels = kmeans.fit_predict(features)
#     return labels

# def extract_keywords(descriptions):
#     """Extract keywords from descriptions."""
#     vectorizer = CountVectorizer(max_features=5, stop_words='english')
#     vectorizer.fit(descriptions)
#     keywords = vectorizer.get_feature_names_out()
#     return keywords

# @app.route('/')
# def index():
#     books = scrape_book_data()
#     image_urls = [book['img_url'] for book in books]
#     images = download_images(image_urls)
#     features = extract_features(images)
#     labels = cluster_images(features)
    
#     # Group books by cluster labels
#     clusters = {i: [] for i in range(max(labels) + 1)}
#     for book, label in zip(books, labels):
#         clusters[label].append(book)

#     # Extract keywords from book titles as a proxy for descriptions
#     titles = [book['title'] for book in books]
#     keywords = extract_keywords(titles)

#     return render_template_string(HTML_TEMPLATE, clusters=clusters, keywords=keywords)

# if __name__ == '__main__':
#     app.run(debug=True)



















from flask import Flask, render_template_string
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Define the HTML template for rendering the clusters with improved styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Book Image Clusters</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f7f7f7;
            margin: 0;
            padding: 0;
        }
        .container {
            width: 80%;
            margin: 20px auto;
            background-color: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 40px;
        }
        .cluster {
            margin-bottom: 50px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }
        .cluster:last-child {
            border-bottom: none;
        }
        .cluster h2 {
            color: #007bff;
            text-align: center;
            margin-bottom: 20px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            grid-gap: 20px;
        }
        .grid img {
            width: 100%;
            border-radius: 5px;
            transition: transform 0.2s;
        }
        .grid img:hover {
            transform: scale(1.05);
        }
        .keywords {
            text-align: center;
            margin-top: 20px;
            color: #555;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Book Image Clusters</h1>
        {% for cluster_id, cluster_data in clusters.items() %}
            <div class="cluster">
                <h2>Cluster {{ cluster_id }}</h2>
                <div class="grid">
                    {% for book in cluster_data['books'] %}
                        <img src="{{ book['img_url'] }}" alt="{{ book['title'] }}">
                    {% endfor %}
                </div>
                <p class="keywords">Keywords: {{ cluster_data['keywords']|join(", ") }}</p>
            </div>
        {% endfor %}
    </div>
</body>
</html>
"""

# The rest of your functions (scrape_book_data, download_images, extract_features, cluster_images) remain the same
def scrape_book_data(base_url="http://books.toscrape.com"):
    """Scrape book titles, images, and descriptions from the given URL."""
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    books = []
    for product in soup.select('article.product_pod'):
        img_element = product.select_one('img.thumbnail')
        title = img_element['alt']
        img_url = base_url + '/' + img_element['src'].replace('..', '')
        # Placeholder for description - this would require scraping individual book pages
        description = 'Description not available'
        books.append({'title': title, 'img_url': img_url})
    return books

def download_images(image_urls):
    """Download images from a list of URLs."""
    images = []
    for url in image_urls:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')  # Convert to RGB
        images.append(img)
    return images

def extract_features(images):
    """Extract features from images using VGG16."""
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

def cluster_images(features, n_clusters=5):
    """Cluster images based on extracted features."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels
def extract_keywords_for_cluster(descriptions):
    """Extract keywords from descriptions for each cluster."""
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(descriptions)
    keywords = vectorizer.get_feature_names_out()
    return keywords

@app.route('/')
def index():
    books = scrape_book_data()
    images = download_images([book['img_url'] for book in books])
    features = extract_features(images)
    labels = cluster_images(features)

    # Group books by cluster labels and extract keywords for each cluster
    clusters = {}
    for book, label in zip(books, labels):
        if label not in clusters:
            clusters[label] = {'books': [], 'descriptions': []}
        clusters[label]['books'].append(book)
        clusters[label]['descriptions'].append(book['title'])  # Assuming titles as descriptions

    # Extract keywords for each cluster
    for cluster_id, cluster_data in clusters.items():
        descriptions = cluster_data['descriptions']
        cluster_data['keywords'] = extract_keywords_for_cluster(descriptions) if descriptions else []

    return render_template_string(HTML_TEMPLATE, clusters=clusters)

if __name__ == '__main__':
    app.run(debug=True)
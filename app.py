import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import pickle
import os
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import base64



# Load Universal Sentence Encoder
@st.cache_resource
def load_use_model():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

use_model = load_use_model()


# Text-Based KNN Model
@st.cache_resource
def load_knn_text():
    return joblib.load("knn_text.pk")  

knn_text = load_knn_text()



# Image-Based KNN Model
@st.cache_resource
def load_knn_image():
    return joblib.load("knn_image.pk")  

knn_image = load_knn_image()



# Image Paths from CSV
@st.cache_resource
def load_image_paths():
    df = pd.read_csv("image_paths.csv")  
    return df["Image Path"].tolist()

image_paths = load_image_paths()



# CNN - Model
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("cnn_model.keras")  

cnn_model = load_cnn_model()




# Function to retrieve images based on text query
def retrieve_images_text(query_text, category=None, max_results=5, knn_depth=10):
    query_vector = np.array(use_model([query_text]))
    distances, indices = knn_text.kneighbors(query_vector, n_neighbors=knn_depth)

    result_images = []
    for i in indices[0]:
        img_path = image_paths[i]

        if category and category.lower() not in img_path.lower():
            continue  # Apply category filter

        result_images.append(img_path)
        if len(result_images) >= max_results:
            break

    return result_images



def load_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize
    return image_array


def extract_image_features(image_path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image = np.expand_dims(np.array(image) / 255.0, axis=0)
    features = cnn_model.predict(image)
    return features.flatten()


# Function to retrieve similar images based on an uploaded image
def retrieve_images_from_image(uploaded_image, category=None, max_results=5, knn_depth=10):
    image_vector = extract_image_features(uploaded_image)  # Convert image to vector
    distances, indices = knn_image.kneighbors([image_vector.flatten()], n_neighbors=knn_depth)

    result_images = []
    for i in indices[0]:
        img_path = image_paths[i]

        # Ensure image belongs to the specified category
        if category and category.lower() not in img_path.lower():
            continue  # Skip images that do not match the category
        
        result_images.append(img_path)
        
        if len(result_images) >= max_results:
            break

    return result_images




# Function to display images in a grid
def display_images(image_list):
    cols = st.columns(3)  # Create a 3-column layout
    
    for i, img_path in enumerate(image_list):
        # Ensure the image path includes "static/"
        img_path = os.path.join("static", img_path) if not img_path.startswith("static/") else img_path

        if os.path.exists(img_path):
            with cols[i % 3]:  # Arrange images in a 3-column layout
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
        else:
            st.error(f"❌ Image not found: {img_path}")



# Streamlit UI
# Function to set the background image
def set_background_image(image_path):
    """
    Set a background image in the Streamlit app using base64 encoding.

    Parameters:
    - image_path: str, path to the image file (e.g., 'background.jpg')
    """
    # Read and encode the image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode()

    # Create the CSS for the background
    background_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    # Inject the CSS into the Streamlit app
    st.markdown(background_css, unsafe_allow_html=True)

# Set the background image
set_background_image("background.png")
st.title("🖼️ Virtual Interior Design Retrieval")

# Text-Based Image Retrieval
st.subheader("🔍 Search Interior Designs by Description")
query_text = st.text_input("Enter a description:")
category_filter = st.selectbox("Select a category:", ["Livingroom", "Bathroom", "Bedroom", "Kitchen","Dinning"], index=0)

if st.button("Show"):
    retrieved_images = retrieve_images_text(query_text, category_filter)  

    if retrieved_images:
        st.success(f"Found {len(retrieved_images)} images!")
        display_images(retrieved_images)
    else:
        st.warning("No matching images found.")


# Image-Based Retrieval
st.subheader("📷 Search Similar Interiors by Image")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
category = st.selectbox("Select category:", ["Livingroom", "Bathroom", "Bedroom", "Kitchen","Dinning"], index=0)


if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Display"):
        retrieved_images = retrieve_images_from_image(uploaded_file, category)
    
        if retrieved_images:
            st.success(f"Found {len(retrieved_images)} images!")
            display_images(retrieved_images)
        else:
            st.warning("No matching images found in the selected category.")

# Upload new images
st.header("📤 Upload Interior Design Images")
uploaded_new_file = st.file_uploader("Upload a new image to store", type=["png", "jpg", "jpeg"])

if uploaded_new_file is not None:
    save_path = os.path.join("static/Interior_images/", uploaded_new_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_new_file.getbuffer())

    st.success(f"Image saved: {save_path}")
    st.image(uploaded_new_file, caption="Uploaded Image", use_container_width=True)

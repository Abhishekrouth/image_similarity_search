import streamlit as st
import requests
from PIL import Image

st.title('Image Similarity Search App')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Image")
    
    if st.button("Find Similar Images"):
        response = requests.post("http://127.0.0.1:5000/search", files={"image": uploaded_file})
        if response.status_code == 200:
            results = response.json().get("top_matches", [])
            for r in results:
                img_path = r["image_path"]  
                score = r["similarity_score"]
                metadata = r["Metadata"]
                st.json(metadata)
                image = Image.open(img_path)
                st.image(image)        
        else:
            st.error("Image not found")

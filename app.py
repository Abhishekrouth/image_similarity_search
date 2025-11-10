from flask import Flask, request, jsonify
import torch
import clip
from PIL import Image
import os, json
import faiss
import numpy as np

app = Flask(__name__)

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

dataset = './dataset/'
embedding_file = 'embeddings.json'
batch = 16
def generate_embeddings():
    embeddings = {}
    if os.path.exists(embedding_file):
        with open(embedding_file, 'r') as f:
            embeddings = json.load(f)
    image_path =[]
    for root, dirs, files in os.walk(dataset):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                path = os.path.join(root, file)
                image_path.append(path)
    for i in range(0, len(image_path), batch):
        batch_files = image_path[i:i+batch]
        
        batch_images = []
        for path in batch_files:
            if path in embeddings:
                continue
            img = preprocess(Image.open(path)).unsqueeze(0)
            batch_images.append(img)

        batch_tensor = torch.cat(batch_images).to(device)
        with torch.no_grad():
            batch_features = model.encode_image(batch_tensor).cpu().numpy()

        for j, path in enumerate(batch_files):               
            if path in embeddings:
                continue
            file = os.path.basename(path)
            metadata = {
                "Image name": file,
                "category": os.path.basename(os.path.dirname(path)),
                "Images": img
            }
            embeddings[path] = {
                "embedding": batch_features[j].tolist(),
                "metadata": metadata
            }
    with open(embedding_file, 'w') as f:
        json.dump(embeddings, f)
if not os.path.exists(embedding_file):
    generate_embeddings()
with open(embedding_file, 'r') as f:
    saved_embeddings = json.load(f)
image_path = list(saved_embeddings.keys())
embedding_matrix = np.array([v["embedding"] for v in saved_embeddings.values()],dtype='float32')
faiss.normalize_L2(embedding_matrix)
index = faiss.IndexFlatIP(512)
index.add(embedding_matrix) 

@app.route('/')
def home():
    return "Image Similarity Search."

@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return jsonify({"error": "Upload image"}), 400

    uploaded_file = request.files['image']
    image = preprocess(Image.open(uploaded_file)).unsqueeze(0).to(device)

    with torch.no_grad():
        input_features = model.encode_image(image).cpu().numpy().astype('float32')
    faiss.normalize_L2(input_features)    
    top_n = 5
    D, I = index.search(input_features, top_n)
    top_matches = []
    for i, score in zip(I[0], D[0]):
        path = image_path[i]
        data = saved_embeddings[path]
        top_matches.append({
            "image_path": path,
            "similarity_score": round(float(score), 3),
            "Metadata": data["metadata"]
        })
    return jsonify({"top_matches": top_matches})

if __name__ == '__main__':
    app.run(debug=True)

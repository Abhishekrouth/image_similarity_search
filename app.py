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

def generate_embeddings():
    embeddings = {}
    for root, dirs, files in os.walk(dataset):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                path = os.path.join(root, file)
                image = preprocess(Image.open(path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = model.encode_image(image).cpu().numpy().flatten().tolist()
                embeddings[path] = features
                

    with open(embedding_file, 'w') as f:
        json.dump(embeddings, f)

if not os.path.exists(embedding_file):
    generate_embeddings()

with open(embedding_file, 'r') as f:
    saved_embeddings = json.load(f)

image_paths = list(saved_embeddings.keys())
embedding_matrix = np.array(list(saved_embeddings.values())).astype('float32')

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

    D, I = index.search(input_features, 3)
    top_matches = []
    for i, score in zip(I[0], D[0]):
        top_matches.append({
            "image": image_paths[i],
            "similarity_score": round(float(score), 3)
        })

    return jsonify({"top_matches": top_matches})


if __name__ == '__main__':
    app.run(debug=True)

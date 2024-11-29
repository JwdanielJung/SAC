import json
import numpy as np
import faiss
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import normalize
import os


class Searcher:
    def __init__(self, embed_model, client=None):
        """
        Initializes the FAISSHandler with the embedding model and optional OpenAI client.
        """
        self.embed_model = embed_model
        self.client = client

    def save_to_faiss(self, embeddings, file_name="train"):
        # Save embeddings to a FAISS index.

        # Normalize embeddings for Inner Product similarity
        embeddings = normalize(np.array(embeddings, dtype=np.float32))

        # Create and populate FAISS index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        # Name the FAISS index file

        faiss_file = f"./data/{file_name}_{self.embed_model}.index"
        faiss.write_index(index, faiss_file)

        print(f"FAISS index saved to {faiss_file}")
        return index

    def similarity_search(self, embeddings, index, file_name="default", top_k=1):
        os.makedirs("results", exist_ok=True)
        file_path = os.path.join("results", file_name)

        # Normalize embeddings for compatibility with FAISS index
        embeddings = normalize(embeddings)
        with open(file_path, "w") as f:
            for id, embedding in enumerate(embeddings):
                similarity, indices = index.search(
                    np.expand_dims(embedding, axis=0), top_k
                )
                result = {
                    "id": id,
                    "similarity": np.mean(similarity, axis=1).tolist()[0],
                    "top_k_index": indices.reshape(-1).tolist(),
                }
                f.write(json.dumps(result) + "\n")

        return pd.read_json(file_path, lines=True)

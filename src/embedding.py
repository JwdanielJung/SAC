from openai import OpenAI
from angle_emb import AnglE
from sentence_transformers import SentenceTransformer
import numpy as np


class Embedding:
    def __init__(self, embed_model, client=None):
        self.embed_model = embed_model
        self.client = client
        self.model = None
        if embed_model == "openai":
            self.model = "text-embedding-3-large"
        elif embed_model == "uae":
            self.model = AnglE.from_pretrained(
                "WhereIsAI/UAE-Large-V1", pooling_strategy="cls"
            ).cuda()
        elif embed_model == "mixbread":
            self.model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

    def _openai_embedding(self, texts):
        def _call_api(inputs):
            embedding = self.client.embeddings.create(input=inputs, model=self.model)
            return embedding

        if len(texts) >= 500:
            chunks = [texts[i : i + 500] for i in range(0, len(texts), 500)]
            embeddings = []
            for chunk in chunks:
                embed = _call_api(chunk)
                temp = [embed.data[i].embedding for i in range(len(chunk))]
                embeddings.extend(temp)
        else:
            embed = _call_api(texts)
            embeddings = [embed.data[i].embedding for i in range(len(texts))]
        return np.array(embeddings, dtype=np.float32)

    def _chunk_input_ids(self, input_ids, tokenizer, chunk_size=200, overlap=50):
        chunks = []
        for i in range(0, len(input_ids), chunk_size - overlap):
            chunk = input_ids[i : i + chunk_size]
            chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
        return chunks

    def _uae_embedding(self, texts):
        embeddings = []
        for text in texts:
            input_ids = self.model.tokenizer.encode(text, add_special_tokens=True)
            if len(input_ids) >= 512:
                chunks = self._chunk_input_ids(input_ids, self.model.tokenizer)
                embeddings.append(np.mean(self.model.encode(chunks), axis=0))
            else:
                embeddings.append(
                    self.model.encode(
                        self.model.tokenizer.decode(input_ids, skip_special_tokens=True)
                    )
                )
        return np.array([embed.reshape(-1) for embed in embeddings], dtype=np.float32)

    def _mixbread_embedding(self, texts):
        embeddings = []
        for text in texts:
            input_ids = self.model.tokenizer.encode(text, add_special_tokens=True)
            if len(input_ids) >= 512:
                chunks = self._chunk_input_ids(input_ids, self.model.tokenizer)
                embeddings.append(np.mean(self.model.encode(chunks), axis=0))
            else:
                embeddings.append(
                    self.model.encode(
                        self.model.tokenizer.decode(input_ids, skip_special_tokens=True)
                    )
                )
        return np.array([embed.reshape(-1) for embed in embeddings], dtype=np.float32)

    def generate_embeddings(self, texts):
        if self.embed_model == "openai":
            return self._openai_embedding(texts)
        elif self.embed_model == "uae":
            return self._uae_embedding(texts)
        elif self.embed_model == "mixbread":
            return self._mixbread_embedding(texts)
        else:
            raise ValueError(f"Unsupported embed_model: {self.embed_model}")

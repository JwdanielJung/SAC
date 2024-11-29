from dotenv import load_dotenv
from openai import OpenAI
from src.preprocessing import preprocessing
from src.embedding import Embedding
from src.search import Searcher
from src.optimize import Optimizer
from src.classifier import Classifier

import faiss
import pandas as pd
import os

if __name__ == "__main__":

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # argparse 할 것들
    is_prepared = True
    embed_model = "openai"  # ["openai", "uae", "mxbai"] choice
    lambda_loss = 0.05
    top_k = 1
    model = "gpt4"  # ["llama_guard", "moderation", "perspective", "gradsafe", "gpt4"] choice

    embedder = Embedding(embed_model, client)
    searcher = Searcher(embed_model, client)

    if not is_prepared:
        train, val, test = preprocessing()
        train_embeddings = embedder.generate_embeddings(train["text"].tolist())
        train_index = searcher.save_to_faiss(train_embeddings)
    else:
        train_index = faiss.read_index(
            f"./data/train_{embed_model}.index"
        )  # faiss 불러오기
        val = pd.read_csv("./data/val.csv")
        test = pd.read_csv("./data/test.csv")

    # optimizing threshold
    val_embeddings = embedder.generate_embeddings(val["text"].tolist())
    val_results = searcher.similarity_search(
        val_embeddings, train_index, file_name=f"val_{embed_model}_{lambda_loss}.jsonl"
    )
    optimizer = Optimizer(
        similarities=val_results["similarity"].tolist(),
        labels=val["label"].tolist(),
        lambda_loss=lambda_loss,
    )
    safe_threshold, unsafe_threshold = optimizer.optimize()

    # filtering
    test_embeddings = embedder.generate_embeddings(test["text"].tolist()[0:3])
    # id: test dataset key & top_k_index: train dataset key
    results = searcher.similarity_search(
        test_embeddings,
        train_index,
        file_name=f"test_{embed_model}_{lambda_loss}.jsonl",
    )

    confi_unsafe = []
    confi_safe = []
    unconfident = []

    loss_unsafe = []
    loss_safe = []

    for id, (similarity, label) in enumerate(
        zip(results["similarity"].tolist(), test["label"].tolist()[0:3])
    ):
        if label == 0 and similarity <= safe_threshold:
            confi_safe.append(id)
        elif label == 1 and similarity >= unsafe_threshold:
            confi_unsafe.append(id)

        elif label == 0 and similarity >= unsafe_threshold:
            loss_safe.append(id)

        elif label == 1 and similarity <= safe_threshold:
            loss_unsafe.append(id)

        else:
            unconfident.append(id)

    # classification
    classifier = Classifier(model, client)
    classifier.classification(texts=test.iloc[unconfident]["text"].tolist())

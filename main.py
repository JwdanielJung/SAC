from src.configs import openai_client, perspective_client, parse_arguments
from src.preprocessing import preprocessing
from src.embedding import Embedding
from src.search import Searcher
from src.optimize import Optimizer
from src.classifier import Classifier
from utils import evaluate

import faiss
import pandas as pd


if __name__ == "__main__":

    args = parse_arguments()

    embedder = Embedding(args.embed_model, openai_client)
    searcher = Searcher(args.embed_model, openai_client)

    if not args.is_prepared:
        train, val, test = preprocessing()
        train_embeddings = embedder.generate_embeddings(train["text"].tolist())
        train_index = searcher.save_to_faiss(train_embeddings)
    else:
        train_index = faiss.read_index(
            f"./data/train_{args.embed_model}.index"
        )  # load faiss index
        val = pd.read_csv("./data/val.csv")
        test = pd.read_csv("./data/test.csv")

    # optimizing threshold
    val_embeddings = embedder.generate_embeddings(val["text"].tolist())
    val_results = searcher.similarity_search(
        val_embeddings,
        train_index,
        file_name=f"val_{args.embed_model}_{args.lambda_loss}.jsonl",
    )
    optimizer = Optimizer(
        similarities=val_results["similarity"].tolist(),
        labels=val["label"].tolist(),
        lambda_loss=args.lambda_loss,
    )
    safe_threshold, unsafe_threshold = optimizer.optimize()

    # filtering
    test_embeddings = embedder.generate_embeddings(test["text"].tolist())
    # id: test dataset key & top_k_index: train dataset key
    results = searcher.similarity_search(
        test_embeddings,
        train_index,
        file_name=f"test_{args.embed_model}_{args.lambda_loss}.jsonl",
    )

    confi_unsafe = []
    confi_safe = []
    unconfident = []

    loss_unsafe = []
    loss_safe = []

    for id, (similarity, label) in enumerate(
        zip(results["similarity"].tolist(), test["label"].tolist())
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
    classifier = Classifier(args.model, openai_client, perspective_client)

    preds = classifier.classification(texts=test.iloc[unconfident]["text"].tolist())
    labels = test.iloc[unconfident]["label"].tolist()

    print(preds, labels)

    print(evaluate(preds, labels))

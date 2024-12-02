from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(preds, labels):
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(
        labels, preds, average="binary" if len(set(labels)) == 2 else "weighted"
    )
    recall = recall_score(
        labels, preds, average="binary" if len(set(labels)) == 2 else "weighted"
    )
    f1 = f1_score(
        labels, preds, average="binary" if len(set(labels)) == 2 else "weighted"
    )

    # Return results as a dictionary
    results = {
        "accuracy": round(float(accuracy), 2),
        "precision": round(float(precision), 2),
        "recall": round(float(recall), 2),
        "f1_score": round(float(f1), 2),
    }
    return results

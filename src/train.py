"""MLOps part: train, evaluate, track (MLflow), version and save the model."""
import json, random, datetime
from pathlib import Path
import joblib, mlflow, pandas as pd
from src.nlp import ItemMatcher


def add_typo(word, rng):
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    kind = rng.choice(["drop", "swap", "dup"])
    if kind == "drop":
        return word[:i] + word[i + 1:]
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + word[i] + word[i:]


def make_queries(products, per_product=6, seed=0):
    rng = random.Random(seed)
    queries = []
    for p in products:
        for _ in range(per_product):
            q = add_typo(p, rng)
            if rng.random() < 0.3:
                q += "s"
            queries.append((q, p))
    return queries


def evaluate(matcher, queries):
    ok = sum(matcher.match(q)[0] == truth for q, truth in queries)
    return ok / len(queries)


def main():
    prices = pd.read_csv("data/prices.csv")
    products = sorted(prices["product"].unique())
    queries = make_queries(products)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("grocery-matcher")
    best = None
    for ngram in [(1, 2), (2, 3), (2, 4), (3, 5)]:
        with mlflow.start_run(run_name=f"ngram_{ngram[0]}_{ngram[1]}"):
            matcher = ItemMatcher(products, ngram_range=ngram)
            acc = evaluate(matcher, queries)
            mlflow.log_param("ngram_range", str(ngram))
            mlflow.log_param("n_products", len(products))
            mlflow.log_metric("accuracy", acc)
            print(f"ngram_range={ngram}  accuracy={acc:.3f}")
            if best is None or acc > best[0]:
                best = (acc, ngram, matcher)

    acc, ngram, matcher = best
    Path("models").mkdir(exist_ok=True)
    joblib.dump(matcher, "models/matcher.joblib")
    meta = {
        "model_version": datetime.datetime.now(datetime.timezone.utc).strftime("v%Y%m%d-%H%M%S"),
        "accuracy": round(acc, 4),
        "best_ngram_range": list(ngram),
        "n_products": len(products),
        "n_test_queries": len(queries),
    }
    json.dump(meta, open("models/metrics.json", "w"), indent=2)
    print("Saved best model:", meta)


if __name__ == "__main__":
    main()

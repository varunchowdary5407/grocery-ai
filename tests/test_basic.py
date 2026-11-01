import json
import pandas as pd
from src.nlp import ItemMatcher, parse_item, split_items
from src.optimize import optimize


def test_split():
    assert split_items("milk, eggs and rice") == ["milk", "eggs", "rice"]


def test_parse():
    assert parse_item("2 kg rice") == (2.0, "kg", "rice")
    assert parse_item("half kg sugar") == (0.5, "kg", "sugar")


def test_typo_matching():
    m = ItemMatcher(["rice", "milk"])
    assert m.match("ricee")[0] == "rice"
    assert m.match("milkk")[0] == "milk"


def test_optimizer():
    prices = pd.DataFrame({"product": ["rice", "rice"], "store": ["A", "B"], "price": [10.0, 8.0]})
    res = optimize({"rice": 2}, prices)
    assert res["best_single_store"] == "B"


def test_model_quality():
    meta = json.load(open("models/metrics.json"))
    assert meta["accuracy"] >= 0.7, "Model accuracy dropped below 70%"

"""Creates a small grocery price dataset (4 stores x 36 products)."""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# (product, category, unit, base price in INR)
PRODUCTS = [
    ("rice", "Grains", "kg", 60), ("wheat flour", "Grains", "kg", 45), ("oats", "Grains", "kg", 150),
    ("milk", "Dairy", "litre", 56), ("curd", "Dairy", "kg", 70), ("butter", "Dairy", "pack", 58),
    ("cheese", "Dairy", "pack", 120), ("paneer", "Dairy", "pack", 90),
    ("eggs", "Protein", "dozen", 84), ("chicken", "Protein", "kg", 220), ("fish", "Protein", "kg", 350),
    ("tomato", "Vegetables", "kg", 30), ("onion", "Vegetables", "kg", 35), ("potato", "Vegetables", "kg", 28),
    ("carrot", "Vegetables", "kg", 45), ("spinach", "Vegetables", "bunch", 20), ("cauliflower", "Vegetables", "kg", 40),
    ("banana", "Fruits", "dozen", 50), ("apple", "Fruits", "kg", 160), ("orange", "Fruits", "kg", 90),
    ("mango", "Fruits", "kg", 120), ("grapes", "Fruits", "kg", 110),
    ("sugar", "Essentials", "kg", 45), ("salt", "Essentials", "kg", 20), ("cooking oil", "Essentials", "litre", 140),
    ("tea powder", "Essentials", "pack", 110), ("coffee", "Essentials", "pack", 180),
    ("toor dal", "Pulses", "kg", 150), ("moong dal", "Pulses", "kg", 130), ("chickpeas", "Pulses", "kg", 110),
    ("bread", "Bakery", "pack", 40), ("biscuits", "Snacks", "pack", 30), ("noodles", "Snacks", "pack", 15),
    ("soap", "Household", "piece", 40), ("detergent", "Household", "kg", 120), ("toothpaste", "Household", "piece", 90),
]
STORES = {"FreshMart": 1.00, "BudgetBazaar": 0.92, "SuperSaver": 0.96, "QuickCart": 1.08}


def main():
    rows = []
    for name, cat, unit, base in PRODUCTS:
        for store, factor in STORES.items():
            price = round(base * factor * np.random.uniform(0.93, 1.07), 2)
            rows.append({"product": name, "category": cat, "unit": unit, "store": store, "price": price})
    Path("data").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("data/prices.csv", index=False)
    print("Saved data/prices.csv with", len(rows), "rows")


if __name__ == "__main__":
    main()

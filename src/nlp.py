"""NLP part: understand a free-text grocery list."""
import re
from sklearn.feature_extraction.text import TfidfVectorizer

UNITS = {
    "kg": "kg", "kgs": "kg", "kilo": "kg", "kilos": "kg",
    "g": "g", "gm": "g", "gms": "g", "gram": "g", "grams": "g",
    "l": "litre", "ltr": "litre", "litre": "litre", "litres": "litre", "liter": "litre", "liters": "litre",
    "ml": "ml", "dozen": "dozen", "pack": "pack", "packs": "pack", "packet": "pack", "packets": "pack",
    "piece": "piece", "pieces": "piece", "pcs": "piece", "bunch": "bunch",
}
NUMBER_WORDS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "half": 0.5}
FILLER = {"of", "some", "the", "and", "please", "need", "want", "buy", "get", "i", "to", "an", "few", "fresh"}


class ItemMatcher:
    """Matches messy text ('tomatoe', 'milkk') to a product in the catalog."""

    def __init__(self, products, ngram_range=(2, 4)):
        self.products = list(products)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range)
        self.X = self.vec.fit_transform(self.products)

    def match(self, text, threshold=0.35):
        sims = (self.X @ self.vec.transform([text]).T).toarray().ravel()
        i = int(sims.argmax())
        if sims[i] >= threshold:
            return self.products[i], float(sims[i])
        return None, float(sims[i])


def split_items(text):
    parts = re.split(r",|;|\n|\+|\band\b", text.lower())
    return [p.strip() for p in parts if p.strip()]


def parse_item(chunk):
    """'2 kg rice' -> (2.0, 'kg', 'rice')"""
    qty, unit, words = None, None, []
    for tok in re.findall(r"\d+\.?\d*|[a-z]+", chunk.lower()):
        if re.fullmatch(r"\d+\.?\d*", tok):
            qty = float(tok)
        elif tok in NUMBER_WORDS and qty is None:
            qty = float(NUMBER_WORDS[tok])
        elif tok in UNITS and unit is None:
            unit = UNITS[tok]
        elif tok in FILLER:
            continue
        else:
            words.append(tok)
    return qty, unit, " ".join(words)


def to_catalog_qty(qty, unit, catalog_unit):
    if qty is None:
        qty = 1.0
    if unit == "g" and catalog_unit == "kg":
        return qty / 1000
    if unit == "ml" and catalog_unit == "litre":
        return qty / 1000
    if unit is None and catalog_unit == "dozen" and qty > 1:
        return qty / 12
    return qty


def process_text(text, matcher, unit_map):
    out = []
    for chunk in split_items(text):
        qty, unit, name = parse_item(chunk)
        if not name:
            continue
        product, score = matcher.match(name)
        q = to_catalog_qty(qty, unit, unit_map.get(product, "")) if product else None
        out.append({"you_typed": chunk, "product": product,
                    "confidence": round(score, 2), "quantity": None if q is None else round(q, 3)})
    return out

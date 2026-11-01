"""Optimization part: find the cheapest way to buy the basket."""
DELIVERY_FEE = {"FreshMart": 40, "BudgetBazaar": 50, "SuperSaver": 30, "QuickCart": 20}


def aggregate(items):
    basket = {}
    for it in items:
        if it["product"]:
            basket[it["product"]] = basket.get(it["product"], 0) + (it["quantity"] or 1)
    return basket


def optimize(basket, prices):
    d = prices[prices["product"].isin(basket)].copy()
    d["qty"] = d["product"].map(basket)
    d["cost"] = (d["price"] * d["qty"]).round(2)

    # Option A: buy everything from ONE store
    single = {s: round(g["cost"].sum() + DELIVERY_FEE.get(s, 0), 2) for s, g in d.groupby("store")}
    best_store = min(single, key=single.get)

    # Option B: buy each item from its cheapest store
    cheapest = d.loc[d.groupby("product")["cost"].idxmin()]
    stores_used = sorted(cheapest["store"].unique())
    split_total = round(cheapest["cost"].sum() + sum(DELIVERY_FEE.get(s, 0) for s in stores_used), 2)

    return {
        "single_totals": single,
        "best_single_store": best_store,
        "best_single_total": single[best_store],
        "worst_single_total": max(single.values()),
        "split_plan": cheapest[["product", "qty", "store", "price", "cost"]].reset_index(drop=True),
        "split_total": split_total,
        "stores_used": stores_used,
    }

"""
Generate a small synthetic retail feedback CSV for development and testing.

Run from the project root:
    python tests/sample_data/generate_sample.py
"""

import random
import csv
from datetime import datetime, timedelta

POSITIVE = [
    "Absolutely love this product! Fast shipping and great quality.",
    "Best purchase I've made this year. Highly recommend!",
    "The quality exceeded my expectations. Will definitely buy again.",
    "Arrived quickly, beautifully packaged. Perfect gift.",
    "Incredible value for money. Customer service was also fantastic.",
    "Works exactly as described. Very happy with my purchase.",
    "Outstanding quality and super fast delivery. 5 stars!",
    "My whole family loves it. Great product at a great price.",
]

NEGATIVE = [
    "Terrible quality. Broke after just one week of use.",
    "Item arrived damaged and the return process was a nightmare.",
    "Not worth the money at all. Very disappointed.",
    "Customer service was unhelpful and rude. Will not buy again.",
    "Completely different from the photos. Misleading description.",
    "Took 3 weeks to arrive and then it was the wrong item.",
    "Poor build quality. Felt cheap and fell apart immediately.",
    "Had to chase the delivery multiple times. Appalling service.",
]

NEUTRAL = [
    "Product is okay. Nothing special but it does the job.",
    "Average quality for the price. Delivery was on time.",
    "It's alright. Not as good as I hoped but acceptable.",
    "Functional product. Exactly what was described.",
    "Decent item. Arrived within the expected timeframe.",
    "Product matches the description. Standard quality.",
]

CATEGORIES = ["Electronics", "Clothing", "Home & Garden", "Toys", "Beauty", "Sports"]


def generate(n=500, output_path="tests/sample_data/retail_reviews.csv"):
    random.seed(42)
    start = datetime(2023, 1, 1)

    rows = []
    for i in range(n):
        roll = random.random()
        if roll < 0.55:
            text, rating = random.choice(POSITIVE), random.randint(4, 5)
        elif roll < 0.80:
            text, rating = random.choice(NEUTRAL), random.randint(3, 3)
        else:
            text, rating = random.choice(NEGATIVE), random.randint(1, 2)

        date = (start + timedelta(days=random.randint(0, 364))).strftime("%Y-%m-%d")
        category = random.choice(CATEGORIES)

        rows.append({
            "review_id":   i + 1,
            "review_text": text,
            "rating":      rating,
            "date":        date,
            "category":    category,
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["review_id", "review_text", "rating", "date", "category"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Generated {n} rows → {output_path}")


if __name__ == "__main__":
    generate()

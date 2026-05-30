import os
import random
import uuid
from datetime import datetime, timedelta

import pandas as pd

RANDOM_STATE = 42
TARGET_SIZE = 10000
SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "amazon_reviews_sample.csv")

CATEGORIES = [
    "Electronics",
    "Books",
    "Home_and_Kitchen",
    "Sports_and_Outdoors",
    "Clothing_Shoes_and_Jewelry",
    "Beauty_and_Personal_Care",
    "Toys_and_Games",
    "Health_and_Household",
]

BRANDS = {
    "Electronics": ["Samsung", "Apple", "Sony", "LG", "Bose", "Anker", "Dell", "HP"],
    "Books": ["Penguin", "HarperCollins", "Simon Schuster", "Oxford", "Penguin Random House"],
    "Home_and_Kitchen": ["KitchenAid", "Ninja", "Instant Pot", "Cuisinart", "Keurig"],
    "Sports_and_Outdoors": ["Nike", "Adidas", "Columbia", "Under Armour", "Fitbit"],
    "Clothing_Shoes_and_Jewelry": ["Levi", "Calvin Klein", "Fossil", "Ray Ban"],
    "Beauty_and_Personal_Care": ["Olay", "Neutrogena", "L Oreal", "Dove", "CeraVe"],
    "Toys_and_Games": ["LEGO", "Hasbro", "Mattel", "Fisher Price"],
    "Health_and_Household": ["Tylenol", "Colgate", "Oral B", "3M", "Bounty"],
}

POSITIVE_TEMPLATES = [
    "I purchased this {product} from {brand} last month and I am extremely satisfied with the quality. "
    "The {feature} works exactly as advertised and exceeded my expectations for daily use at home. "
    "Setup was straightforward and the instructions were clear. Customer support from {brand} was helpful when I had a question. "
    "Compared to similar items in the {category} category, this offers excellent value at ${price}. "
    "I would highly recommend it to anyone looking for reliable performance and durable materials.",
    "After two weeks of regular use, this {product} has proven to be one of my best purchases on Amazon. "
    "The build quality feels premium and the {feature} is responsive and consistent. "
    "Shipping was fast and the package arrived intact. I chose {brand} because of positive reviews and they delivered. "
    "For ${price}, you get features that competitors charge much more for. Easy to install and works great with my other devices.",
    "Excellent product overall. The {feature} on this {brand} {product} is impressive and saves me time every day. "
    "Battery life and durability are better than I expected. I use it in my office and at home without issues. "
    "Five stars because it solved the problem I had with my old unit. Worth every penny at ${price}.",
]

NEGATIVE_TEMPLATES = [
    "I regret buying this {product} from {brand}. The {feature} stopped working after only ten days of light use. "
    "I contacted support but received no useful response. For ${price}, I expected much better quality in the {category} department. "
    "The item arrived with scratches and the packaging was damaged. I requested a refund because it does not match the description online. "
    "Many reviewers warned about defects and I should have listened. Very disappointed with this purchase.",
    "Terrible experience with this {brand} product. The {feature} is unreliable and makes loud noises during operation. "
    "I tried troubleshooting for hours but nothing fixed the issue. At ${price}, this is overpriced junk. "
    "I returned it and am waiting for my refund. Do not buy unless you enjoy wasting money on broken electronics.",
    "One star because the {product} failed immediately. The {feature} never worked properly out of the box. "
    "Cheap materials and poor design from {brand}. Customer service was unhelpful and slow. "
    "Save yourself the hassle and buy a different brand in {category}. I will not purchase from {brand} again.",
]

NEUTRAL_EXTRA = [
    "The product is okay but nothing special. It does the basic job for ${price}.",
    "Average quality for the {category} segment. I might upgrade later.",
]

PRODUCTS = {
    "Electronics": ["wireless earbuds", "bluetooth speaker", "laptop stand", "USB hub", "smart watch", "tablet case"],
    "Books": ["mystery novel", "cookbook", "history guide", "science textbook", "biography"],
    "Home_and_Kitchen": ["blender", "coffee maker", "air fryer", "knife set", "food processor"],
    "Sports_and_Outdoors": ["running shoes", "yoga mat", "water bottle", "camping tent", "resistance bands"],
    "Clothing_Shoes_and_Jewelry": ["winter jacket", "running shorts", "leather belt", "sunglasses"],
    "Beauty_and_Personal_Care": ["moisturizer", "shampoo set", "electric toothbrush", "face serum"],
    "Toys_and_Games": ["building blocks set", "board game", "action figure", "puzzle"],
    "Health_and_Household": ["vitamin pack", "paper towels", "hand sanitizer", "first aid kit"],
}

FEATURES = [
    "battery life",
    "wireless connectivity",
    "noise cancellation",
    "touchscreen",
    "motor",
    "heating element",
    "water resistance",
    "storage capacity",
]

CATEGORY_KEYWORDS = {
    "Electronics": "The bluetooth chipset firmware update and HDMI port worked flawlessly for my home office setup.",
    "Books": "The paperback binding spine chapters narrative pacing made this novel engaging for weekend reading.",
    "Home_and_Kitchen": "The kitchen countertop appliance dishwasher safe nonstick coating performed well during meal prep.",
    "Sports_and_Outdoors": "These running shoes trail grip moisture wicking fabric held up on long outdoor workouts.",
    "Clothing_Shoes_and_Jewelry": "The fabric fit waistband stitching style held up after multiple washes and daily wear.",
    "Beauty_and_Personal_Care": "My skincare routine serum moisturizer texture absorption improved after two weeks of use.",
    "Toys_and_Games": "Kids enjoyed the puzzle pieces board game rules family game night lasted hours.",
    "Health_and_Household": "Household supplies packaging dosage instructions storage shelf life met our family needs.",
}


def make_review(category, sentiment, rng):
    brand = rng.choice(BRANDS[category])
    product = rng.choice(PRODUCTS[category])
    feature = rng.choice(FEATURES)
    price = rng.randint(15, 299)
    if sentiment == "positivo":
        rating = rng.choice([4, 5, 5, 5])
        body = rng.choice(POSITIVE_TEMPLATES).format(
            product=product, brand=brand, feature=feature, category=category.replace("_", " "), price=price
        )
        headline = rng.choice([
            f"Great {product}!",
            f"Love my new {brand} purchase",
            f"Highly recommend this {product}",
            f"Excellent value at ${price}",
        ])
    else:
        rating = rng.choice([1, 2, 2])
        body = rng.choice(NEGATIVE_TEMPLATES).format(
            product=product, brand=brand, feature=feature, category=category.replace("_", " "), price=price
        )
        headline = rng.choice([
            f"Disappointed with {product}",
            f"Do not buy this {brand} item",
            f"Returned defective {product}",
            f"Waste of money at ${price}",
        ])
    if rng.random() < 0.15:
        body += " " + rng.choice(NEUTRAL_EXTRA).format(category=category.replace("_", " "), price=price)
    body += " " + CATEGORY_KEYWORDS[category]
    return headline, body, rating, brand, product, price


def main():
    rng = random.Random(RANDOM_STATE)
    rows = []
    per_cat = TARGET_SIZE // len(CATEGORIES)
    start_date = datetime(2022, 1, 1)
    for category in CATEGORIES:
        for sentiment in ["positivo", "negativo"]:
            n = per_cat // 2
            for _ in range(n):
                headline, body, rating, brand, product, price = make_review(category, sentiment, rng)
                review_date = (start_date + timedelta(days=rng.randint(0, 900))).strftime("%Y-%m-%d")
                texto = f"{headline} {body}"
                rows.append({
                    "marketplace": "US",
                    "review_id": str(uuid.uuid4()),
                    "product_id": f"B{rng.randint(10000000, 99999999)}",
                    "product_title": f"{brand} {product.title()}",
                    "product_category": category,
                    "categoria_top": category,
                    "star_rating": rating,
                    "sentimento": sentiment,
                    "verified_purchase": rng.choice(["Y", "Y", "Y", "N"]),
                    "review_headline": headline,
                    "review_body": body,
                    "texto_completo": texto,
                    "review_date": review_date,
                    "n_palavras": len(texto.split()),
                })
    df = pd.DataFrame(rows)
    df = df.sample(n=TARGET_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    os.makedirs(os.path.dirname(SAMPLE_PATH), exist_ok=True)
    df.to_csv(SAMPLE_PATH, index=False)
    print(f"Gerado: {SAMPLE_PATH} ({len(df)} linhas)")
    print(df["product_category"].value_counts())
    print(df["sentimento"].value_counts())


if __name__ == "__main__":
    main()

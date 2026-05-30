import glob
import os
import re

import numpy as np
import pandas as pd

RANDOM_STATE = 42
MIN_WORDS = 30
TARGET_MIN_WORDS = 100
PREFERRED_MIN_WORDS = 110
TARGET_SIZE = 10000
MIN_PER_CATEGORY = 200
TOP_CATEGORIES = 8
TOP_PRODUCTS = 80
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
SAMPLE_DIR = os.path.join(BASE_DIR, "data", "sample")
SAMPLE_PATH = os.path.join(SAMPLE_DIR, "amazon_reviews_sample.csv")
FINE_FOOD_PATH = os.path.join(SAMPLE_DIR, "Reviews.csv")
SKIP_NAMES = {"amazon_reviews_sample.csv"}


def find_raw_file():
    if os.path.isfile(FINE_FOOD_PATH):
        return FINE_FOOD_PATH
    patterns = ["*.tsv", "*.csv", "**/*.tsv", "**/*.csv"]
    for pattern in patterns:
        matches = glob.glob(os.path.join(RAW_DIR, pattern), recursive=True)
        if matches:
            return matches[0]
    for pattern in ["*.csv", "*.tsv"]:
        for path in glob.glob(os.path.join(SAMPLE_DIR, pattern)):
            if os.path.basename(path) not in SKIP_NAMES:
                return path
    return None


def word_count(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\b\w+\b", text))


def clean_html(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def assign_sentiment(rating):
    if rating >= 4:
        return "positivo"
    if rating <= 2:
        return "negativo"
    return None


def normalize_columns(df):
    rename_map = {
        "Review Headline": "review_headline",
        "Review Body": "review_body",
        "Star Rating": "star_rating",
        "Product Title": "product_title",
        "Product Category": "product_category",
        "Review Date": "review_date",
        "Verified Purchase": "verified_purchase",
        "Summary": "review_headline",
        "Text": "review_body",
        "Score": "star_rating",
        "Id": "review_id",
        "ProductId": "product_id",
        "Time": "review_time_unix",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def add_metadata_defaults(df):
    if "marketplace" not in df.columns:
        df["marketplace"] = "US"
    if "verified_purchase" not in df.columns:
        df["verified_purchase"] = ""
    if "review_date" not in df.columns and "review_time_unix" in df.columns:
        ts = pd.to_numeric(df["review_time_unix"], errors="coerce")
        df["review_date"] = pd.to_datetime(ts, unit="s", errors="coerce").dt.strftime("%Y-%m-%d")
    if "product_title" not in df.columns:
        df["product_title"] = "Product " + df["product_id"].astype(str)
    return df


def assign_product_groups(df, top_products):
    ranked = list(top_products)
    groups = np.array_split(ranked, TOP_CATEGORIES)
    mapping = {}
    for idx, chunk in enumerate(groups):
        label = f"Gourmet_Foods_G{idx + 1}"
        for pid in chunk:
            mapping[pid] = label
    df = df[df["product_id"].isin(ranked)].copy()
    df["product_category"] = df["product_id"].map(mapping)
    df["categoria_top"] = df["product_category"]
    return df


def assign_marketplace_categories(df):
    if "product_category" in df.columns and df["product_category"].notna().sum() > len(df) * 0.5:
        if "categoria_top" not in df.columns:
            df["categoria_top"] = df["product_category"]
        return df
    if "product_id" not in df.columns:
        raise ValueError("Dataset sem product_category nem product_id.")
    counts = df["product_id"].value_counts()
    top_products = counts.head(TOP_PRODUCTS).index.tolist()
    return assign_product_groups(df, top_products)


def length_pool(df):
    for threshold in (PREFERRED_MIN_WORDS, TARGET_MIN_WORDS, MIN_WORDS):
        pool = df[df["n_palavras"] >= threshold].copy()
        if len(pool) >= TARGET_SIZE:
            return pool, threshold
    return df[df["n_palavras"] >= MIN_WORDS].copy(), MIN_WORDS


def weighted_sample(subset, n):
    if len(subset) <= n:
        return subset
    w = subset["n_palavras"].astype(float) ** 2
    w = w / w.sum()
    return subset.sample(n=n, random_state=RANDOM_STATE, weights=w)


def stratified_sample(df, target_size):
    cats = df["categoria_top"].value_counts()
    cats = cats[cats >= 1].head(TOP_CATEGORIES).index.tolist()
    df = df[df["categoria_top"].isin(cats)].copy()
    per_group = max(1, target_size // (len(cats) * 2))
    parts = []
    for cat in cats:
        for sent in ["positivo", "negativo"]:
            subset = df[(df["categoria_top"] == cat) & (df["sentimento"] == sent)]
            if subset.empty:
                continue
            n = min(len(subset), per_group)
            parts.append(weighted_sample(subset, n))
    sample = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(sample) < target_size and "review_id" in df.columns:
        taken = set(sample["review_id"].astype(str))
        remaining = df[~df["review_id"].astype(str).isin(taken)]
        if not remaining.empty:
            extra_n = min(len(remaining), target_size - len(sample))
            sample = pd.concat(
                [sample, weighted_sample(remaining, extra_n)], ignore_index=True
            )
    if len(sample) > target_size:
        sample = weighted_sample(sample, target_size)
    return sample.reset_index(drop=True)


def build_sample(df):
    df = normalize_columns(df)
    df = add_metadata_defaults(df)
    df["review_headline"] = df.get("review_headline", "").fillna("").astype(str).map(clean_html)
    df["review_body"] = df.get("review_body", "").fillna("").astype(str).map(clean_html)
    df["texto_completo"] = (df["review_headline"] + " " + df["review_body"]).str.strip()
    df["n_palavras"] = df["texto_completo"].apply(word_count)
    df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce")
    df = df.dropna(subset=["star_rating"])
    df["star_rating"] = df["star_rating"].astype(int)
    df["sentimento"] = df["star_rating"].apply(assign_sentiment)
    df = df.dropna(subset=["sentimento"])
    if "product_id" in df.columns:
        df["product_id"] = df["product_id"].astype(str)
    df, used_min = length_pool(df)
    df = assign_marketplace_categories(df)
    df, used_min = length_pool(df)
    sample = stratified_sample(df, TARGET_SIZE)
    cols = [
        "marketplace",
        "review_id",
        "product_id",
        "product_title",
        "product_category",
        "categoria_top",
        "star_rating",
        "sentimento",
        "verified_purchase",
        "review_headline",
        "review_body",
        "texto_completo",
        "review_date",
        "n_palavras",
    ]
    keep = [c for c in cols if c in sample.columns]
    return sample[keep], used_min


def read_raw_csv(raw_path):
    sep = "\t" if raw_path.endswith(".tsv") else ","
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(raw_path, sep=sep, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(
        raw_path, sep=sep, low_memory=False, encoding="latin-1", encoding_errors="replace"
    )


def main():
    raw_path = find_raw_file()
    if raw_path is None:
        raise FileNotFoundError(
            "Coloque Reviews.csv em data/sample/ ou o dataset do Kaggle em data/raw/."
        )
    print(f"Carregando: {raw_path}")
    df = read_raw_csv(raw_path)
    print(f"Linhas brutas: {len(df)}")
    sample, used_min = build_sample(df)
    os.makedirs(os.path.dirname(SAMPLE_PATH), exist_ok=True)
    sample.to_csv(SAMPLE_PATH, index=False)
    print(f"Amostra salva: {SAMPLE_PATH} ({len(sample)} linhas)")
    print(f"Filtro de comprimento: >= {used_min} palavras")
    print(f"Média de palavras: {sample['n_palavras'].mean():.1f}")
    print(f"Mediana: {sample['n_palavras'].median():.0f}")
    print(sample["categoria_top"].value_counts())
    print(sample["sentimento"].value_counts())


if __name__ == "__main__":
    main()

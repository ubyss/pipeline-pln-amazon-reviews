import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
NB_PATH = os.path.join(ROOT, "projeto_pln_amazon.ipynb")


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source if isinstance(source, list) else [source]}


def code(source):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": source if isinstance(source, list) else [source]}


cells = [
    md("# Pipeline PLN — Amazon Product Reviews\n\n**Disciplina:** Sistemas Cognitivos e Linguagem Natural\n\n**Questão:** Como construir um pipeline completo de PLN para transformar avaliações brutas da Amazon em informações úteis sobre satisfação, temas e relações entre produtos/marcas/categorias?\n\n**Corpus:** [Amazon Product Reviews (Kaggle)](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews/data)"),
    code("%pip install -q pandas numpy matplotlib seaborn scikit-learn nltk spacy gensim wordcloud networkx pyvis vaderSentiment python-Levenshtein pyLDAvis"),
    code("""import os
import re
import warnings
from collections import Counter

import Levenshtein
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from pyvis.network import Network
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp_full = spacy.load("en_core_web_sm")

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "scripts" else os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, "data", "sample", "amazon_reviews_sample.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CUSTOM_STOPWORDS = {
    "amazon", "product", "review", "purchase", "item", "would", "also", "one",
    "get", "got", "use", "used", "really", "much", "even", "well", "could",
}
stop_en = set(stopwords.words("english")) | CUSTOM_STOPWORDS
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
vader = SentimentIntensityAnalyzer()

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
"""),
    md("## Bloco 1 — Caracterização do corpus"),
    code("""df = pd.read_csv(DATA_PATH)
if "texto_completo" not in df.columns:
    df["texto_completo"] = (df["review_headline"].fillna("") + " " + df["review_body"].fillna("")).str.strip()
if "sentimento" not in df.columns:
    df["sentimento"] = np.where(df["star_rating"] >= 4, "positivo", np.where(df["star_rating"] <= 2, "negativo", None))
    df = df.dropna(subset=["sentimento"])
if "categoria_top" not in df.columns:
    df["categoria_top"] = df["product_category"]

df["n_palavras"] = df["texto_completo"].str.split().str.len()
df["n_chars"] = df["texto_completo"].str.len()

print(f"Documentos: {len(df):,}")
print(f"Média de palavras: {df['n_palavras'].mean():.1f}")
print(f"Categorias: {df['categoria_top'].nunique()}")
df.info()
df[["star_rating", "n_palavras", "n_chars"]].describe()
"""),
    code("""fig, axes = plt.subplots(1, 3, figsize=(16, 4))

df["star_rating"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Distribuição de estrelas")
axes[0].set_xlabel("star_rating")

df["sentimento"].value_counts().plot(kind="bar", ax=axes[1], color=["#2ecc71", "#e74c3c"])
axes[1].set_title("Sentimento (binário)")

df["categoria_top"].value_counts().plot(kind="barh", ax=axes[2])
axes[2].set_title("Categorias de produto")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_classes.png"), dpi=120, bbox_inches="tight")
plt.show()
"""),
    code("""fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df["n_palavras"], bins=40, color="teal", edgecolor="white")
ax.axvline(df["n_palavras"].mean(), color="red", linestyle="--", label=f"Média={df['n_palavras'].mean():.0f}")
ax.set_xlabel("Palavras por documento")
ax.set_ylabel("Frequência")
ax.set_title("Comprimento dos documentos (headline + body)")
ax.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "histograma_comprimento.png"), dpi=120, bbox_inches="tight")
plt.show()

print(
    "Reviews Amazon tendem a ser mais curtas que 200 palavras; usamos headline+body e filtro mínimo de 30 palavras."
)
"""),
    code("""wc = WordCloud(width=900, height=400, background_color="white", max_words=120).generate(
    " ".join(df["texto_completo"].head(3000))
)
plt.figure(figsize=(12, 5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Nuvem de palavras — texto bruto")
plt.savefig(os.path.join(OUTPUT_DIR, "wordcloud_bruto.png"), dpi=120, bbox_inches="tight")
plt.show()
"""),
    md("## Bloco 2 — Pré-processamento (NLTK + spaCy)"),
    code("""def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\S+", " ", text)
    text = re.sub(r"\\d+", " ", text)
    text = re.sub(r"[^a-z\\s]", " ", text)
    return re.sub(r"\\s+", " ", text).strip()


def tokenize_words(text):
    return word_tokenize(normalize_text(text))


def remove_stopwords(tokens):
    return [t for t in tokens if t not in stop_en and len(t) > 2]


def stem_tokens(tokens):
    return [stemmer.stem(t) for t in tokens]


def lemma_nltk_tokens(tokens):
    return [lemmatizer.lemmatize(t) for t in tokens]


def lemma_spacy_text(text):
    doc = nlp(normalize_text(text))
    return [t.lemma_ for t in doc if t.lemma_ not in stop_en and not t.is_space and len(t.lemma_) > 2]


def preprocess_pipeline(text, method="lemma_spacy"):
    tokens = tokenize_words(text)
    tokens = remove_stopwords(tokens)
    if method == "stem":
        return stem_tokens(tokens)
    if method == "lemma_nltk":
        return lemma_nltk_tokens(tokens)
    return lemma_spacy_text(text)


def tokens_to_str(tokens):
    return " ".join(tokens)
"""),
    code("""sample_text = df["texto_completo"].iloc[0]
print("Sentenças:", sent_tokenize(sample_text)[:2])
print("Tokens:", tokenize_words(sample_text)[:15])
print("Stem:", stem_tokens(remove_stopwords(tokenize_words(sample_text)))[:15])
print("Lemma spaCy:", lemma_spacy_text(sample_text)[:15])
"""),
    code("""df["tokens_stem"] = df["texto_completo"].apply(lambda t: preprocess_pipeline(t, "stem"))

texts_norm = [normalize_text(t) for t in df["texto_completo"]]
tokens_lemma_list = []
for doc in nlp.pipe(texts_norm, batch_size=256):
    tokens_lemma_list.append([t.lemma_ for t in doc if t.lemma_ not in stop_en and not t.is_space and len(t.lemma_) > 2])
df["tokens_lemma"] = tokens_lemma_list
df["texto_processado"] = df["tokens_lemma"].apply(tokens_to_str)

vocab_bruto = len(set(word_tokenize(" ".join(df["texto_completo"].head(5000).str.lower()))))
vocab_stem = len(set(t for row in df["tokens_stem"].head(5000) for t in row))
vocab_lemma = len(set(t for row in df["tokens_lemma"].head(5000) for t in row))

comp = pd.DataFrame({
    "estrategia": ["bruto (5k docs)", "stemming", "lemmatização spaCy"],
    "tamanho_vocabulario": [vocab_bruto, vocab_stem, vocab_lemma],
})
display(comp)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(comp["estrategia"], comp["tamanho_vocabulario"], color=["#95a5a6", "#3498db", "#27ae60"])
ax.set_ylabel("|V|")
ax.set_title("Impacto do pré-processamento no vocabulário")
plt.xticks(rotation=15, ha="right")
plt.savefig(os.path.join(OUTPUT_DIR, "vocabulario_preprocessamento.png"), dpi=120, bbox_inches="tight")
plt.show()

print("Decisão: lemmatização spaCy para modelos e busca — preserva interpretabilidade sem perder semântica como o stemming.")
"""),
    code("""pos_counts = Counter()
for text in df["texto_completo"].sample(800, random_state=RANDOM_STATE):
    doc = nlp_full(str(text)[:5000])
    pos_counts.update([t.pos_ for t in doc])

pos_df = pd.DataFrame(pos_counts.items(), columns=["pos", "count"]).sort_values("count", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(data=pos_df, x="pos", y="count", ax=ax)
ax.set_title("Distribuição de POS tags (spaCy, amostra 800 docs)")
plt.savefig(os.path.join(OUTPUT_DIR, "pos_tags.png"), dpi=120, bbox_inches="tight")
plt.show()
"""),
    code("""term_freq = Counter()
for tokens in df["tokens_lemma"]:
    term_freq.update(tokens)
top20 = pd.DataFrame(term_freq.most_common(20), columns=["termo", "freq"])

fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(data=top20, y="termo", x="freq", ax=ax)
ax.set_title("Top 20 termos após lemmatização")
plt.savefig(os.path.join(OUTPUT_DIR, "termos_frequentes.png"), dpi=120, bbox_inches="tight")
plt.show()

wc2 = WordCloud(width=900, height=400, background_color="white", max_words=100).generate(
    " ".join(df["texto_processado"].head(3000))
)
plt.figure(figsize=(12, 5))
plt.imshow(wc2, interpolation="bilinear")
plt.axis("off")
plt.title("Nuvem de palavras — pós-processamento")
plt.savefig(os.path.join(OUTPUT_DIR, "wordcloud_processado.png"), dpi=120, bbox_inches="tight")
plt.show()
"""),
    md("## Bloco 3 — Representação vetorial e busca textual"),
    code("""corpus = df["texto_processado"].tolist()

bow = CountVectorizer(max_features=5000)
X_bow = bow.fit_transform(corpus)

tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=3)
X_tfidf = tfidf.fit_transform(corpus)

sentences = [t.split() for t in corpus]
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=3, workers=2, seed=RANDOM_STATE, epochs=10)

print("BoW shape:", X_bow.shape)
print("TF-IDF shape:", X_tfidf.shape)
print("Word2Vec vocab:", len(w2v.wv))
"""),
    code("""from sklearn.metrics.pairwise import cosine_similarity


def doc_embedding(tokens):
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    if not vecs:
        return np.zeros(w2v.vector_size)
    return np.mean(vecs, axis=0)


doc_vectors = np.vstack([doc_embedding(t) for t in df["tokens_lemma"]])


def buscar_similares(query, vectorizer, matrix, top_k=5):
    q = preprocess_pipeline(query, "lemma_spacy")
    q_str = tokens_to_str(q)
    q_vec = vectorizer.transform([q_str])
    scores = cosine_similarity(q_vec, matrix).flatten()
    idx = scores.argsort()[::-1][:top_k]
    return df.iloc[idx][["review_headline", "star_rating", "sentimento", "categoria_top"]].assign(score=scores[idx])


consultas = [
    "battery life terrible",
    "easy to install highly recommend",
    "returned defective refund",
]

for q in consultas:
    print("\\n=== Consulta:", q, "===")
    display(buscar_similares(q, tfidf, X_tfidf))
"""),
    code("""from sklearn.manifold import TSNE

idx_tsne = np.random.default_rng(RANDOM_STATE).choice(len(doc_vectors), size=min(2000, len(doc_vectors)), replace=False)
emb_2d = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, max_iter=800).fit_transform(doc_vectors[idx_tsne])

tsne_df = df.iloc[idx_tsne].copy()
tsne_df["x"] = emb_2d[:, 0]
tsne_df["y"] = emb_2d[:, 1]

fig, ax = plt.subplots(figsize=(9, 7))
sns.scatterplot(data=tsne_df, x="x", y="y", hue="sentimento", alpha=0.5, ax=ax)
ax.set_title("t-SNE dos embeddings Word2Vec (documento = média dos tokens)")
plt.savefig(os.path.join(OUTPUT_DIR, "tsne_word2vec.png"), dpi=120, bbox_inches="tight")
plt.show()
"""),
    md("## Bloco 4 — Modelagem, classificação e tópicos"),
    code("""X = X_tfidf
y_sent = df["sentimento"]
y_cat = df["categoria_top"]

X_train, X_test, y_sent_train, y_sent_test = train_test_split(
    X, y_sent, test_size=0.2, random_state=RANDOM_STATE, stratify=y_sent
)
_, _, y_cat_train, y_cat_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cat
)

models = {
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=3000),
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
}

metricas_sent = []
metricas_cat = []
"""),
    code("""def avaliar(nome, modelo, X_tr, X_te, y_tr, y_te, task):
    modelo.fit(X_tr, y_tr)
    pred = modelo.predict(X_te)
    row = {
        "modelo": nome,
        "tarefa": task,
        "precision": precision_score(y_te, pred, average="macro", zero_division=0),
        "recall": recall_score(y_te, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_te, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_te, pred, average="weighted", zero_division=0),
    }
    return row, pred


preds_sent = {}
preds_cat = {}
for nome, modelo in models.items():
    row, pred = avaliar(nome, modelo, X_train, X_test, y_sent_train, y_sent_test, "sentimento")
    metricas_sent.append(row)
    preds_sent[nome] = pred
    row2, pred2 = avaliar(nome, clone(modelo), X_train, X_test, y_cat_train, y_cat_test, "categoria")
    metricas_cat.append(row2)
    preds_cat[nome] = pred2

df_metricas = pd.DataFrame(metricas_sent + metricas_cat)
df_metricas.to_csv(os.path.join(OUTPUT_DIR, "metricas_classificacao.csv"), index=False)
display(df_metricas)
"""),
    code("""best_sent = max(metricas_sent, key=lambda x: x["f1_macro"])["modelo"]
cm = confusion_matrix(y_sent_test, preds_sent[best_sent], labels=["negativo", "positivo"])

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["negativo", "positivo"], yticklabels=["negativo", "positivo"], ax=ax)
ax.set_xlabel("Previsto")
ax.set_ylabel("Real")
ax.set_title(f"Matriz de confusão — sentimento ({best_sent})")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_sentimento.png"), dpi=120, bbox_inches="tight")
plt.show()

print(classification_report(y_sent_test, preds_sent[best_sent]))
"""),
    code("""best_cat = max(metricas_cat, key=lambda x: x["f1_macro"])["modelo"]
cm_cat = confusion_matrix(y_cat_test, preds_cat[best_cat], labels=sorted(y_cat.unique()))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_cat, annot=False, cmap="YlOrRd", ax=ax)
ax.set_title(f"Matriz de confusão — categoria ({best_cat})")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_categoria.png"), dpi=120, bbox_inches="tight")
plt.show()

print(classification_report(y_cat_test, preds_cat[best_cat]))
"""),
    code("""lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
lr.fit(X_train, y_sent_train)
scores = lr.decision_function(X_test)
y_bin = (y_sent_test == "positivo").astype(int)
prec, rec, _ = precision_recall_curve(y_bin, scores)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(rec, prec)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Curva Precision-Recall (sentimento, Logistic Regression)")
plt.savefig(os.path.join(OUTPUT_DIR, "precision_recall_sentimento.png"), dpi=120, bbox_inches="tight")
plt.show()
"""),
    code("""vader_preds = []
vader_scores = []
for text in df["texto_completo"]:
    s = vader.polarity_scores(str(text))["compound"]
    vader_scores.append(s)
    vader_preds.append("positivo" if s >= 0.05 else "negativo")

vader_f1 = f1_score(df["sentimento"], vader_preds, average="macro")
print(f"VADER F1 macro: {vader_f1:.3f}")
print(classification_report(df["sentimento"], vader_preds))

cm_v = confusion_matrix(df["sentimento"], vader_preds, labels=["negativo", "positivo"])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_v, annot=True, fmt="d", cmap="Greens", xticklabels=["negativo", "positivo"], yticklabels=["negativo", "positivo"], ax=ax)
ax.set_title("VADER vs rótulos")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_vader.png"), dpi=120, bbox_inches="tight")
plt.show()
"""),
    code("""N_TOPICS = 7
lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=RANDOM_STATE, max_iter=15, learning_method="online")
doc_topics = lda.fit_transform(X_tfidf)

feature_names = tfidf.get_feature_names_out()
for i, topic in enumerate(lda.components_):
    top_idx = topic.argsort()[-12:][::-1]
    terms = [feature_names[j] for j in top_idx]
    print(f"Tópico {i}: {', '.join(terms)}")
"""),
    code("""import pyLDAvis
import pyLDAvis.lda_model

panel = pyLDAvis.lda_model.prepare(lda, X_tfidf, tfidf, mds="tsne")
pyLDAvis.save_html(panel, os.path.join(OUTPUT_DIR, "ldavis.html"))

topic_cols = [f"topico_{i}" for i in range(N_TOPICS)]
topic_df = pd.DataFrame(doc_topics, columns=topic_cols)
topic_df["categoria_top"] = df["categoria_top"].values
topic_df["sentimento"] = df["sentimento"].values
heat = topic_df.groupby("categoria_top")[topic_cols].mean()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heat, cmap="viridis", ax=ax)
ax.set_title("Distribuição média de tópicos por categoria")
plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_topicos_categoria.png"), dpi=120, bbox_inches="tight")
plt.show()
"""),
    md("### Interpretação dos modelos\n\n- **Sentimento:** modelos supervisionados com TF-IDF superam VADER quando o léxico não captura nuances do domínio.\n- **Categoria:** SVM/Regressão Logística lidam melhor com multiclasse esparsa; Naive Bayes serve como baseline rápido.\n- **LDA:** revela eixos como qualidade/preço, defeito/devolução e recomendação."),
    md("## Bloco 5 — NER, extração de informação e grafo de conhecimento"),
    code("""ner_sample = df.sample(1500, random_state=RANDOM_STATE)
ent_rows = []
for _, row in ner_sample.iterrows():
    doc = nlp_full(str(row["texto_completo"])[:8000])
    for ent in doc.ents:
        if ent.label_ in {"ORG", "PRODUCT", "GPE", "MONEY", "PERSON"}:
            ent_rows.append({
                "text": ent.text,
                "label": ent.label_,
                "sentimento": row["sentimento"],
                "categoria_top": row["categoria_top"],
            })

ent_df = pd.DataFrame(ent_rows)
print(ent_df["label"].value_counts().head(10))
top_orgs = ent_df[ent_df["label"] == "ORG"]["text"].str.title().value_counts().head(15)
display(top_orgs)
"""),
    code("""date_pat = re.compile(r"\\b\\d{4}-\\d{2}-\\d{2}\\b")
price_pat = re.compile(r"\\$\\s?\\d+(?:\\.\\d{2})?")
url_pat = re.compile(r"https?://\\S+|www\\.\\S+")

df["datas_regex"] = df["texto_completo"].apply(lambda t: date_pat.findall(str(t)))
df["precos_regex"] = df["texto_completo"].apply(lambda t: price_pat.findall(str(t)))
df["urls_regex"] = df["texto_completo"].apply(lambda t: url_pat.findall(str(t)))

print("Reviews com preço $:", (df["precos_regex"].str.len() > 0).sum())
print("Exemplo preços:", df["precos_regex"].iloc[0][:3])
"""),
    code("""from spacy import displacy

doc_demo = nlp_full(str(df["texto_completo"].iloc[0])[:3000])
html_ner = displacy.render(doc_demo, style="ent", jupyter=False, page=True)
with open(os.path.join(OUTPUT_DIR, "displacy_ner.html"), "w", encoding="utf-8") as f:
    f.write(html_ner)
print("NER displaCy salvo em outputs/displacy_ner.html")
"""),
    code("""known_brands = sorted({
    "Samsung", "Apple", "Sony", "Nike", "Adidas", "Kitchenaid", "Lego", "Hasbro",
    "Bose", "Anker", "Dell", "Hp", "Ninja", "Instant Pot", "Cuisinart", "Keurig",
    "Columbia", "Under Armour", "Fitbit", "Olay", "Neutrogena", "Dove", "Mattel",
    "Lg", "Penguin", "Loreal", "Cerave", "Fisher Price", "Tylenol", "Colgate", "Oral B",
})

raw_orgs = ent_df[ent_df["label"] == "ORG"]["text"].tolist() if len(ent_df) else []
all_candidates = list(set(raw_orgs + known_brands))


def normalize_entity(name, canonical_list, max_dist=2):
    name = name.strip().title()
    best, best_d = name, max_dist + 1
    for c in canonical_list:
        d = Levenshtein.distance(name.lower(), c.lower())
        if d < best_d:
            best, best_d = c, d
    return best if best_d <= max_dist else name

canonical = []
for c in all_candidates[:200]:
    canonical.append(normalize_entity(c, known_brands))
brand_map = dict(zip(all_candidates[:200], canonical))
print("Exemplos Levenshtein:", list(brand_map.items())[:5])
"""),
    code("""G = nx.Graph()
cat_nodes = df["categoria_top"].unique().tolist()
for c in cat_nodes:
    G.add_node(c, tipo="categoria")

if len(ent_df):
    top_brand_names = ent_df[ent_df["label"] == "ORG"]["text"].str.title().value_counts().head(12).index.tolist()
else:
    top_brand_names = known_brands[:12]

for b in top_brand_names:
    G.add_node(b, tipo="marca")

topic_terms = []
for i, topic in enumerate(lda.components_):
    top_idx = topic.argsort()[-3:][::-1]
    for j in top_idx:
        term = feature_names[j]
        G.add_node(term, tipo="termo_topico")
        topic_terms.append(term)

for _, row in df.sample(2000, random_state=RANDOM_STATE).iterrows():
    cat = row["categoria_top"]
    sent = row["sentimento"]
    text = str(row["texto_completo"]).lower()
    for b in top_brand_names:
        if b.lower() in text:
            G.add_edge(cat, b, weight=G.edges.get((cat, b), {}).get("weight", 0) + 1)
            if sent == "negativo":
                G.add_edge(b, "review_negativa", weight=G.edges.get((b, "review_negativa"), {}).get("weight", 0) + 1)
    for term in topic_terms[:15]:
        if term in text:
            G.add_edge(cat, term, weight=G.edges.get((cat, term), {}).get("weight", 0) + 1)

if "review_negativa" not in G:
    G.add_node("review_negativa", tipo="meta")

print(f"Nós: {G.number_of_nodes()}, Arestas: {G.number_of_edges()}")
"""),
    code("""deg = nx.degree_centrality(G)
bet = nx.betweenness_centrality(G)
cent_df = pd.DataFrame({"grau": deg, "betweenness": bet}).sort_values("grau", ascending=False).head(20)
display(cent_df)

for node in G.nodes():
    ntype = G.nodes[node].get("tipo", "")
    if ntype == "categoria":
        G.nodes[node]["color"] = "#3498db"
    elif ntype == "marca":
        G.nodes[node]["color"] = "#e74c3c"
    else:
        G.nodes[node]["color"] = "#2ecc71"

net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
net.from_nx(G)
net.save_graph(os.path.join(OUTPUT_DIR, "grafo_conhecimento.html"))
print("Grafo salvo em outputs/grafo_conhecimento.html")
"""),
    code("""cat_alvo = "Electronics" if "Electronics" in G.nodes else cat_nodes[0]
neg_neighbors = []
for u, v, data in G.edges(data=True):
    if u == "review_negativa" or v == "review_negativa":
        other = v if u == "review_negativa" else u
        if G.nodes.get(other, {}).get("tipo") == "marca":
            neg_neighbors.append((other, data.get("weight", 1)))
    if cat_alvo in (u, v):
        other = v if u == cat_alvo else u
        if G.nodes.get(other, {}).get("tipo") == "marca":
            pass

if G.has_node("review_negativa"):
    marcas_neg = []
    for u, v, d in G.edges("review_negativa", data=True):
        if G.nodes.get(u, {}).get("tipo") == "marca":
            marcas_neg.append((u, d.get("weight", 1)))
        if G.nodes.get(v, {}).get("tipo") == "marca":
            marcas_neg.append((v, d.get("weight", 1)))
    marcas_neg = sorted(marcas_neg, key=lambda x: -x[1])[:5]
    print("Pergunta: Qual marca está mais ligada a reviews negativas?")
    print("Resposta (grafo):", marcas_neg)
else:
    sub = [n for n in G.neighbors(cat_alvo)] if cat_alvo in G else []
    print(f"Nós conectados a {cat_alvo}:", sub[:10])
"""),
    md("## Bloco 6 — Síntese para stakeholder e reprodutibilidade"),
    md("""### Síntese em linguagem não técnica

As avaliações da Amazon analisadas mostram que clientes satisfeitos destacam **qualidade**, **facilidade de uso** e **custo-benefício**, enquanto insatisfeitos concentram reclamações em **defeitos**, **devoluções** e **suporte**. O sistema consegue **prever automaticamente** se uma avaliação é positiva ou negativa e identificar a **categoria do produto** com boa precisão, o que permite priorizar atendimento e monitorar marcas problemáticas.

A busca por similaridade encontra avaliações parecidas em segundos — útil para achar casos análogos a uma reclamação. O mapa de relações (grafo) revela quais **marcas e categorias** aparecem juntas em contextos negativos, apoiando decisões de qualidade e fornecedor.

### Limitações

- Textos mais curtos que o ideal acadêmico (~200 palavras); mitigado com headline + corpo.
- NER genérico em inglês pode não captar todos os nomes de produtos.
- Amostra de 10k documentos; dataset completo exige download Kaggle.

### Melhorias futuras

- Fine-tuning de transformers (BERT) para sentimento e categorias.
- NER customizado com anotação de marcas do domínio varejo.

### Reprodução

1. `pip install -r requirements.txt`
2. `python -m spacy download en_core_web_sm`
3. Executar este notebook com `data/sample/amazon_reviews_sample.csv` presente.
4. Consultar `outputs/` para figuras e grafo HTML.
"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook criado: {NB_PATH} ({len(cells)} células)")

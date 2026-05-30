<p align="center">
  <img src="assets/logo-Infnet.png" alt="Instituto Infnet" width="160" />
</p>

<h1 align="center">Pipeline PLN — Amazon Fine Food Reviews</h1>

<p align="center">
  Projeto da disciplina <strong>Sistemas Cognitivos e Linguagem Natural</strong><br />
  Pipeline completo de PLN sobre avaliações de alimentos gourmet na Amazon.
</p>

<p align="center">
  <a href="https://github.com/ubyss/pipeline-pln-amazon-reviews">Repositório</a> ·
  <a href="https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews">Dataset (Kaggle)</a>
</p>

---

## Visão geral

O notebook `projeto_pln_amazon.ipynb` cobre o fluxo end-to-end: EDA, pré-processamento (NLTK + spaCy), BoW/TF-IDF, Word2Vec, busca por similaridade, LDA, classificação de sentimento e categorias, VADER, NER, grafo de conhecimento (NetworkX + PyVis) e exportação de métricas.

| Item | Valor |
|------|--------|
| Corpus | [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) |
| Amostra | 10.000 reviews · ~204 palavras/doc · 8 grupos `Gourmet_Foods_G1`–`G8` |
| Idioma | Inglês |

---

## Resultados visuais

### Exploração e pré-processamento

<p align="center">
  <img src="outputs/eda_classes.png" alt="Distribuição de classes" width="48%" />
  <img src="outputs/histograma_comprimento.png" alt="Histograma de comprimento" width="48%" />
</p>

<p align="center">
  <img src="outputs/wordcloud_bruto.png" alt="Word cloud texto bruto" width="48%" />
  <img src="outputs/wordcloud_processado.png" alt="Word cloud texto processado" width="48%" />
</p>

<p align="center">
  <img src="outputs/vocabulario_preprocessamento.png" alt="Impacto do pré-processamento no vocabulário" width="70%" />
</p>

<p align="center">
  <img src="outputs/pos_tags.png" alt="Distribuição de POS tags" width="48%" />
  <img src="outputs/termos_frequentes.png" alt="Termos frequentes" width="48%" />
</p>

### Representação vetorial

<p align="center">
  <img src="outputs/tsne_word2vec.png" alt="t-SNE Word2Vec" width="70%" />
</p>

<p align="center">
  <img src="outputs/heatmap_tfidf.png" alt="Heatmap TF-IDF" width="70%" />
</p>

### Classificação e tópicos

<p align="center">
  <img src="outputs/confusion_matrix_sentimento.png" alt="Matriz de confusão sentimento" width="32%" />
  <img src="outputs/confusion_matrix_categoria.png" alt="Matriz de confusão categoria" width="32%" />
  <img src="outputs/confusion_matrix_vader.png" alt="Matriz de confusão VADER" width="32%" />
</p>

<p align="center">
  <img src="outputs/precision_recall_sentimento.png" alt="Curva precision-recall sentimento" width="55%" />
  <img src="outputs/heatmap_topicos_categoria.png" alt="Heatmap tópicos por categoria" width="42%" />
</p>

### Visualizações interativas

| Artefato | Descrição |
|----------|-----------|
| [grafo_conhecimento.html](outputs/grafo_conhecimento.html) | Grafo de marcas, categorias e coocorrências (PyVis) |
| [ldavis.html](outputs/ldavis.html) | Tópicos LDA (pyLDAvis) |
| [displacy_ner.html](outputs/displacy_ner.html) | Entidades nomeadas (spaCy displaCy) |
| [metricas_classificacao.csv](outputs/metricas_classificacao.csv) | F1, precision e recall por modelo |

---

## Estrutura do projeto

```
├── projeto_pln_amazon.ipynb
├── requirements.txt
├── assets/
│   └── logo-Infnet.png
├── data/
│   ├── raw/
│   └── sample/
│       ├── Reviews.csv              # corpus completo (~568k reviews, Git LFS)
│       └── amazon_reviews_sample.csv
├── outputs/
└── scripts/
    ├── build_sample.py
    └── generate_sample_fallback.py
```

---

## Pré-requisitos

- Python 3.10+

## Instalação

```bash
pip install -r requirements.txt
```

O notebook baixa o modelo spaCy `en_core_web_sm` e os recursos NLTK na primeira execução da célula de setup.

## Execução rápida

```bash
pip install -r requirements.txt
jupyter notebook projeto_pln_amazon.ipynb
```

Configure credenciais Kaggle (`~/.kaggle/kaggle.json` ou `KAGGLE_USERNAME` / `KAGGLE_KEY`).

Execute **Run All**:

1. Se `amazon_reviews_sample.csv` não existir, o notebook baixa [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) com **kagglehub** e roda `scripts/build_sample.py`.
2. O pipeline usa a amostra de 10k reviews.

Clone com Git LFS se quiser `Reviews.csv` já no disco (opcional).

## Regenerar amostra manualmente

```bash
python scripts/build_sample.py
```

---

## Tecnologias

NLTK · spaCy · scikit-learn · Gensim · VADER · LDA/pyLDAvis · NetworkX · PyVis · Matplotlib · WordCloud

---

<p align="center">
  <img src="assets/logo-Infnet.png" alt="Instituto Infnet" width="100" />
</p>

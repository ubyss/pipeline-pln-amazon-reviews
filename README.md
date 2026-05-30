# Pipeline PLN — Amazon Product Reviews

Projeto da disciplina Sistemas Cognitivos e Linguagem Natural: pipeline completo de PLN sobre avaliações de produtos Amazon.

## Estrutura

```
├── projeto_pln_amazon.ipynb    # Notebook principal (executar end-to-end)
├── requirements.txt
├── data/
│   ├── raw/                    # Dataset completo do Kaggle (não versionado)
│   └── sample/                 # Amostra reprodutível (~10k reviews)
├── outputs/                    # Figuras, métricas, grafo HTML, LDAvis
├── relatorio/                  # Fonte do relatório técnico (Markdown → PDF)
└── scripts/
    └── build_sample.py         # Gera amostra a partir do dataset bruto
```

## Pré-requisitos

- Python 3.10+
- Conta Kaggle (opcional, se quiser regenerar a amostra a partir do dataset completo)

## Instalação

```bash
pip install -r requirements.txt
```

O notebook baixa automaticamente o modelo spaCy `en_core_web_sm` e os recursos NLTK na primeira execução da célula de setup.

## Execução rápida (sem Kaggle)

O arquivo `data/sample/amazon_reviews_sample.csv` já contém uma amostra estratificada. Basta abrir e executar o notebook:

```bash
jupyter notebook projeto_pln_amazon.ipynb
```

Execute **Run All** na ordem das células (a célula de setup instala o modelo spaCy se ainda não existir).

## Regenerar amostra a partir do Kaggle

1. Baixe o corpus e coloque em `data/sample/Reviews.csv` (formato **Amazon Fine Food Reviews**: colunas `Summary`, `Text`, `Score`, `ProductId`) ou use o dataset multi-categoria em `data/raw/`.

2. Gere a amostra estratificada (~10k reviews, média ~200 palavras):

```bash
python scripts/build_sample.py
```

O script lê `data/sample/Reviews.csv` com prioridade, filtra reviews mais longas (≥110 palavras), agrupa os 80 produtos mais avaliados em **8 categorias** (`Gourmet_Foods_G1` … `G8`) e grava `data/sample/amazon_reviews_sample.csv`.

Alternativa sem Kaggle (apenas demonstração):

```bash
python scripts/generate_sample_fallback.py
```

## Dataset

- **Fonte principal:** [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) (`Reviews.csv`)
- **Amostra do projeto:** `data/sample/amazon_reviews_sample.csv` (10.000 documentos, 8 grupos de produtos, inglês)
- **Colunas:** `review_headline`, `review_body`, `star_rating`, `product_id`, `product_category`, `categoria_top`, `texto_completo`, `review_date`

## Relatório PDF

Exporte o Markdown em `relatorio/` para PDF:

```bash
pandoc relatorio/nome_sobrenome_sistemas-cognitivos-linguagem-natural_pln.md -o relatorio/nome_sobrenome_sistemas-cognitivos-linguagem-natural_pln.pdf
```

Ou use a extensão Markdown PDF do VS Code / imprima do Jupyter.

Substitua `nome_sobrenome` pelo seu nome antes da entrega.

## Saídas geradas

Após executar o notebook, `outputs/` conterá:

- Figuras PNG (EDA, métricas, t-SNE, heatmap TF-IDF, heatmaps de confusão/tópicos)
- `grafo_conhecimento.html` (PyVis)
- `ldavis.html` (pyLDAvis)
- `metricas_classificacao.csv`

# Diagnóstico de Câncer de Mama com Machine Learning

Este projeto desenvolve modelos preditivos para classificar diagnósticos de câncer de mama como malignos ou benignos, utilizando o dataset Breast Cancer Wisconsin (Diagnostic). Com foco em minimizar falsos negativos, implementamos um pipeline completo de análise de dados, seleção de características e modelagem.

## Objetivo

Desenvolver um modelo preditivo robusto e interpretável que ajude na detecção precoce de câncer de mama, priorizando o Recall** para reduzir diagnósticos incorretos de casos malignos.

## Técnicas Utilizadas

- **Seleção de Variáveis**
  - Seleção por correlação
  - SelectKBest
  - Redução de dimensionalidade com PCA

- **Modelos Avaliados**
  - Regressão Logística
  - Random Forest
  - XGBoost (com e sem otimização de hiperparâmetros)
  - Support Vector Machine
  - K-Nearest Neighbors
- **Validação**
  - `StratifiedKFold` com validação cruzada

## Principais Resultados

Modelo escolhido: **XGBoost com hiperparâmetros otimizados**

| Métrica   | Valor   |
|-----------|---------|
| Recall    | 0.9767  |
| Precisão  | 1.0000  |
| F1-Score  | 0.9882  |
| AUC       | 0.9964  |

Detecção eficaz de praticamente todos os casos malignos  
Nenhum falso positivo  
Estabilidade entre treino e validação cruzada

## Estrutura do Projeto

```
breast-cancer-classification/
├── data/
│   └── breast cancer kaggle.csv
├── notebooks/
│   └── breast_cancer.ipynb
├── src/
│   ├── feature_selection.py
│   ├── model_evaluation.py
│   ├── models.py
│   └── utils.py
├── README.md
└── requirements.txt
```

## Dataset

- **Fonte**: [Kaggle - Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Descrição**: Dados clínicos de exames de mama com rótulo binário (Maligno ou Benigno)

## Como Reproduzir

1. Clone o repositório:
```bash
git clone https://github.com/seuusuario/breast-cancer-ml.git
cd breast-cancer-ml
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute os notebooks na pasta `notebooks/` para reproduzir os experimentos.

## Lições Aprendidas

- A importância de priorizar a métrica certa conforme o contexto (neste caso, Recall).
- Como a seleção de variáveis pode impactar significativamente o desempenho dos modelos.
- O papel dos hiperparâmetros no refinamento do modelo.
- A importância da validação cruzada para evitar overfitting.

> Desenvolvido por [Bruno Casini]  
> Contato: [kzini1701@gmail.com]
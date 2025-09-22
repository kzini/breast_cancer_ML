# Diagnóstico de Câncer de Mama com Machine Learning

Desenvolver um modelo preditivo robusto e interpretável para classificar diagnósticos de câncer de mama 
como malignos ou benignos, com foco em minimizar falsos negativos. Para isso, foi implementado um pipeline 
completo de análise de dados, seleção de características e modelagem.

## Técnicas Utilizadas

- **Seleção de Variáveis**
  - Seleção por correlação
  - SelectKBest
  - Redução de dimensionalidade com PCA

- **Modelos Avaliados**
  - Regressão Logística
  - Random Forest
  - XGBoost (com e sem otimização de hiperparâmetros)
  - SVC
  - KNN

- **Validação**
  - StratifiedKFold com validação cruzada

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

## Decisão sobre balanceamento de classes

O dataset tem distribuição relativamente equilibrada (~63% vs 37%), então não apliquei técnicas de 
balanceamento no projeto final. Para lidar com o desbalanceamento moderado, utilizei class_weight='balanced' 
apenas em modelos sensíveis, garantindo ponderação das classes sem alterar os dados reais. Experimentos com 
undersampling e oversampling geraram overfitting, enquanto o uso de SMOTE não trouxe ganhos significativos nas métricas.

## Outras tentativas de aumentar o recall

BalancedBaggingClassifier – sem melhorias.
Threshold Moving – conseguiu aumentar o recall, mas o trade-off com as demais métricas não compensou.

## Preparação para Deploy

O modelo final foi encapsulado em um Pipeline com pré-processamento padronizado e treinado novamente no conjunto de treino. 
Ele foi salvo em models/xgboost_breast_cancer_fs_optimized.pkl e é carregado pelo aplicativo interativo (app.py).

## Estrutura do projeto

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
├── app.py
├── README.md
└── requirements.txt
```

## Dataset

- **Fonte**: [Kaggle - Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Descrição**: Dados clínicos de exames de mama com rótulo binário (Maligno ou Benigno)

## Como Reproduzir

1. Clone o repositório:
```bash
git clone https://github.com/kzini/breast-cancer-ml.git
cd breast-cancer-ml
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute o notebook na pasta `notebooks/` para reproduzir os experimentos.

## Lições Aprendidas

- A importância de priorizar a métrica certa conforme o contexto (neste caso, Recall).
- Como a seleção de variáveis pode impactar significativamente no desempenho dos modelos.
- O papel dos hiperparâmetros no refinamento do modelo.
- A importância da validação cruzada para evitar overfitting.
- Estratégias para lidar com balanceamento de classes e entender quando são realmente necessárias.

> Desenvolvido por Bruno Casini  
> Contato: kzini1701@gmail.com

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, recall_score, precision_score, 
                           f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_pipeline(model, scaling=True):
    """Cria pipeline com ou sem etapa de scaling"""
    steps = []
    if scaling:
        steps.append(('scaler', StandardScaler()))
    steps.append(('classifier', model))
    
    return Pipeline(steps)

def evaluate_model(model, X, y, cv=10, scaling=True):
    pipeline = make_pipeline(model, scaling=scaling)
    predictions = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)
    report_dict = classification_report(y, predictions, output_dict=True)
    
    return pd.DataFrame(report_dict).transpose()

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
    plt.show()

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model, model_name):
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    acc = round(accuracy_score(y_test, y_pred), 4)
    rec = round(recall_score(y_test, y_pred, average='binary'), 4)
    prec = round(precision_score(y_test, y_pred, average='binary'), 4)
    f1 = round(f1_score(y_test, y_pred, average='binary'), 4)
    auc = round(roc_auc_score(y_test, y_pred_prob), 4)

    metrics_df = pd.DataFrame([[model_name, acc, rec, prec, f1, auc]],
                              columns=['Modelo', 'Accuracy', 'Recall', 'Precision', 'F1 Score', 'AUC'])
    
    print(f"\n=== {model_name} ===")
    print("Matriz de Confusão:")
    plot_confusion_matrix(y_test, y_pred)
    
    return metrics_df, model


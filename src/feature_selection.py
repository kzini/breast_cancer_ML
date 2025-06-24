import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

def feature_selection_rfe_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "recall",
    cv_splits: int = 10,
    cv_repeats: int = 3,
    rfe_cv: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[list, Pipeline]:

    xgb_rfe = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    xgb_final = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)

    rfe = RFECV(estimator=xgb_rfe, scoring=scoring, cv=rfe_cv, n_jobs=n_jobs)

    pipeline = Pipeline([
        ('feature_selection', rfe),
        ('classifier', xgb_final)
    ])

    cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=random_state)
    n_scores = cross_val_score(pipeline, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs, error_score='raise')
    pipeline.fit(X, y)
    selected_features = X.columns[rfe.support_].tolist()

    if verbose:
        print(f"{scoring.capitalize()}: {np.mean(n_scores):.3f} (± {np.std(n_scores):.3f})")
        print("Variáveis selecionadas:")
        print(selected_features)
        print(f"\nNúmero de variáveis: {len(selected_features)}")

    return selected_features, pipeline

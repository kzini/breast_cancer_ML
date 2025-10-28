import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def show_basic_info(df: pd.DataFrame):
    print("Shape:", df.shape)
    
    print("\nInfo:")
    print(df.info())

    print("\nDescribe:")
    print(df.describe())
    
    print("\nMissing values:")
    print(df.isnull().sum())

def plot_correlation_heatmap(
    df: pd.DataFrame,
    figsize=(15, 15),
    annot=True,
    fmt='.1f',
    cmap='coolwarm',
    linewidths=0.5
):
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), annot=annot, fmt=fmt, cmap=cmap, linewidths=linewidths)
    plt.tight_layout()
    plt.show()

def normalize_data(X):
    return (X - X.mean()) / X.std()

def plot_violin_features(X, y, start, end, figsize=(15, 15)):
    X_scaled = normalize_data(X)
    data = pd.concat([y, X_scaled.iloc[:, start:end]], axis=1)
    data = pd.melt(data, id_vars="diagnosis", var_name="features", value_name="value")
    
    plt.figure(figsize=figsize)
    sns.violinplot(x="features", y="value", hue="diagnosis", data=data, split=True, inner="quart")
    plt.xticks(rotation=80)
    plt.tight_layout()
    plt.show()

def plot_swarm_features(X, y, start, end, figsize=(15, 15), size=3):
    X_scaled = normalize_data(X)
    data = pd.concat([y, X_scaled.iloc[:, start:end]], axis=1)
    data = pd.melt(data, id_vars="diagnosis", var_name="features", value_name="value")
    plt.figure(figsize=figsize)
    sns.swarmplot(x="features", y="value", hue="diagnosis", data=data, size=size)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

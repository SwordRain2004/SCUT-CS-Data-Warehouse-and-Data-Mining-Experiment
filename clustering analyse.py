"""
    本代码使用BIRCH, K-means, Agglomerative Clustering以及AutoEncoder+K-means
    对Adult数据集进行聚类分析，计算ARI和NMI值并对数据进行PCA降维可视化
"""

"""--------------------------------设置随机种子（可选）--------------------------------"""
import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

"""---------------------------------读取数据并预处理---------------------------------"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_and_preprocess_data(path='./adult/adult.data'):
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    df = pd.read_csv(path, names=column_names, skipinitialspace=True)
    df = df.drop(columns=['education'])
    df = df.dropna()
    labels = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0).values
    df = df.drop(columns=['income'])
    numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_features = [col for col in df.columns if col not in numeric_features]
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    X = preprocessor.fit_transform(df)
    return X.toarray(), labels


X, y_true = load_and_preprocess_data()

"""---------------------------------聚类算法运行函数---------------------------------"""
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from datetime import datetime
from time import time


def evaluate_clustering(model, X, y_true, name=""):
    print(f"Model: {name}")
    start_time = time()
    print(f"Start time: {datetime.now()}")
    y_pred = model.fit_predict(X)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    print(f"End time: {datetime.now()}")
    print(f"Time consuming: {time() - start_time}")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Info (NMI): {nmi:.4f}")
    print()


"""-------------------------------深度学习聚类自编码器实现-------------------------------"""
# import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans
import matplotlib.pyplot as plt


class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_autoencoder(X, epochs=20, batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    model = AutoEncoder(input_dim=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_tensor.size(0))
        for i in range(0, X_tensor.size(0), batch_size):
            indices = perm[i:i + batch_size]
            batch = X_tensor[indices]
            encoded, decoded = model(batch)
            loss = criterion(decoded, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        losses.append(loss.item())
    model.eval()
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(1, 51)), losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    with torch.no_grad():
        encoded_features, _ = model(X_tensor)
    return encoded_features.cpu().numpy()


algorithms = {
    "BIRCH": Birch(n_clusters=2),
    "KMeans": KMeans(n_clusters=2, random_state=42),
    "Agglomerative": AgglomerativeClustering(n_clusters=2)
}
for algorithm in algorithms:
    evaluate_clustering(algorithms[algorithm], X, y_true, algorithm)
encoded_X = train_autoencoder(X, epochs=50)
evaluate_clustering(KMeans(n_clusters=2, random_state=42), encoded_X, y_true, "AutoEncoder + KMeans")

"""---------------------------------数据可视化处理---------------------------------"""
from sklearn.decomposition import PCA


def visualize_cluster_results(X_original, encoded_X, clustering_models, encoded_model, title_suffix=""):
    pca_original = PCA(n_components=2)
    X_pca = pca_original.fit_transform(X_original)
    pca_encoded = PCA(n_components=2)
    encoded_pca = pca_encoded.fit_transform(encoded_X)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.ravel()

    for idx, (name, model) in enumerate(clustering_models.items()):
        labels = model.fit_predict(X_original)
        axs[idx].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=10)
        axs[idx].set_title(f"{name} Clustering")
        axs[idx].set_xlabel("PCA 1")
        axs[idx].set_ylabel("PCA 2")

    encoded_labels = encoded_model.fit_predict(encoded_X)
    axs[3].scatter(encoded_pca[:, 0], encoded_pca[:, 1], c=encoded_labels, cmap='viridis', s=10)
    axs[3].set_title("AutoEncoder + KMeans Clustering")
    axs[3].set_xlabel("PCA 1")
    axs[3].set_ylabel("PCA 2")

    plt.suptitle(f"Clustering Results on Adult Dataset {title_suffix}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


visualize_cluster_results(X, encoded_X, algorithms, KMeans(n_clusters=2, random_state=42))

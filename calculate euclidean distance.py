"""
    本代码利用numpy计算欧几里得距离并排序得到前3名的算法和第1名的数据库
"""

import json
import numpy as np


def algorithm_find():
    # 读取 name vector.txt 文件中的姓名向量
    with open(f"name vector.txt", "r", encoding="utf-8") as f:
        content = f.read()
    content = json.loads(content)
    data = json.loads(content)
    name_embeddings = data["data"][0]["embedding"]
    # 读取 algorithm vector.txt 文件中的算法的向量组
    with open(f"algorithm vector.txt", "r", encoding="utf-8") as f:
        content = f.read()
    content = json.loads(content)
    data = json.loads(content)
    algorithm_embeddings = [item["embedding"] for item in data["data"]]
    Euclidean_Distance = []
    for i, vector in enumerate(algorithm_embeddings):
        Euclidean_Distance.append((i, np.linalg.norm(np.array(name_embeddings) - np.array(vector))))
    Euclidean_Distance.sort(key=lambda x: x[1])
    print(f"与你的姓名的向量的欧几里得距离最接近的算法的向量是第"
          f"{Euclidean_Distance[0][0]}个、第{Euclidean_Distance[1][0]}个、第{Euclidean_Distance[2][0]}个（从0开始计数）")
    for i in Euclidean_Distance:
        print(i)
    """
    分类任务算法：
    (1). 集成学习中的Adaboost
    (2). 朴素贝叶斯(Naive Bayes)
    (3). 决策树C4.5(Decision Trees)
    (4). 集成学习中的 Gradient Tree Boosting
    (5). 支持向量机(Support Vector Machine)
    (6). 最近邻分类器(Nearest Neighbors)
    (7). 集成学习(Ensemble Methods)中的随机森林(Random Forest)
    (8). 分类与回归树CART
    聚类任务算法：
    (1). Affinity Propagation亲和力传播
    (2). BIRCH桦木
    (3). DBSCAN基于密度的聚类算法
    (4). Hierarchical clustering层次聚类
    (5). K-means K均值聚类
    (6). Mean Shift均值位移
    (7). OPTICS光学
    (8). Spectral clustering谱聚类算法
    """


def database_find():
    # 读取 name vector.txt文件中的姓名向量
    with open(f"name vector.txt", "r", encoding="utf-8") as f:
        content = f.read()
    content = json.loads(content)
    data = json.loads(content)
    name_embeddings = data["data"][0]["embedding"]
    # 读取uci数据库中文名的向量
    vectors = []
    for i in range(0, 678, 10):
        with open(f"output{i}-{min(i + 10, 677)}.txt", "r", encoding="utf-8") as f:
            content = f.read()
        content = json.loads(content)
        data = json.loads(content)
        embeddings = [item["embedding"] for item in data["data"]]
        vectors.extend(embeddings)
    min_index = -1
    min_Euclidean_Distance = float('inf')
    Euclidean_Distance = []
    for i, vector in enumerate(vectors):
        Euclidean_Distance.append([i, np.linalg.norm(np.array(name_embeddings) - np.array(vector))])
        if Euclidean_Distance[-1][-1] < min_Euclidean_Distance:
            min_index = i
            min_Euclidean_Distance = Euclidean_Distance[-1][-1]
    print(f"与你的姓名的向量的欧几里得距离最接近的数据库的向量是第{min_index}个（从0开始计数）")  # 175
    for i in sorted(Euclidean_Distance, key=lambda x: x[1]):
        print(i)

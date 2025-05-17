"""
    本代码通过阿里云百炼的模型text-embedding-v3计算得到自己的姓名、算法的中文、数据库的中文名的向量
"""

from openai import OpenAI
import pandas as pd
import json


# 获取自己姓名的向量
def get_name_vector():
    client = OpenAI(
        api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # 输入你自己的API Key
        # 获取API Key教程https://help.aliyun.com/zh/model-studio/get-api-key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
    )

    completion = client.embeddings.create(
        model="text-embedding-v3",
        input='蔡徐坤',  # 输入自己的姓名
        dimensions=1024,
        encoding_format="float"
    )
    with open(f"name vector.txt", "w", encoding="utf-8") as f:
        json.dump(completion.model_dump_json(), f, ensure_ascii=False, indent=2)


# 获取各个算法的中文名的向量
def get_algorithm_vector():
    client = OpenAI(
        api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # 输入你自己的API Key
        # 获取API Key教程https://help.aliyun.com/zh/model-studio/get-api-key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
    )
    algorithm_name = ["亲和力传播", '桦木', '基于密度的聚类算法', '层次聚类', 'K均值聚类', '均值位移', '光学',
                      '谱聚类算法']
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
    completion = client.embeddings.create(
        model="text-embedding-v3",
        input=algorithm_name,
        encoding_format="float"
    )
    # 将转换好的词向量保存到txt文件中
    with open(f"algorithm vector.txt", "w", encoding="utf-8") as f:
        json.dump(completion.model_dump_json(), f, ensure_ascii=False, indent=2)


# 获取678个uci数据库中文名的向量
def get_database_vector():
    file_path = "uci_datasets.xlsx"
    df = pd.read_excel(file_path)

    db_names = df["数据库名字"].dropna().astype(str).tolist()

    client = OpenAI(
        api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # 输入你自己的API Key
        # 获取API Key教程https://help.aliyun.com/zh/model-studio/get-api-key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
    )
    # 由于模型一次最多接收10个字段，因此需要分批次转换
    for i in range(0, len(db_names), 10):
        print(i)
        completion = client.embeddings.create(
            model="text-embedding-v3",
            input=db_names[i:min(i + 10, len(db_names))],
            encoding_format="float"
        )
        # 将转换好的词向量保存到txt文件中
        with open(f"output{i}-{min(i + 10, len(db_names) - 1)}.txt", "w", encoding="utf-8") as f:
            json.dump(completion.model_dump_json(), f, ensure_ascii=False, indent=2)

# SCUT-CS-Data-Warehouse-and-Data-Mining-Experiment
华南理工大学计科全英创新班2025年春数据仓库与数据挖掘实验

希望大家能多多star~

## 实验前置——任务算法与数据集的确定：使用中文 Word Embedding 选取与自己姓名最接近的(使用2-范数)三个算法以及数据集

（1）运行get uci database names.py文件，获取uci数据库的中英文名称，在代码中使用了爬虫技术以及百度翻译的api

（2）运行calculate embedding vector.py文件，计算姓名、算法中文名、数据库中文名的向量，在代码中使用了阿里云百炼的text-embedding-v3模型的api将中文文本向量化

（3）运行calculate enclidean distance.py文件，计算欧几里得距离（即2-范数），确定所需要实现的算法和使用的数据库

## 聚类实验：

我需要使用的数据集为Adult，所需要使用的算法是BIRCH, K-means以及Hierarchical clustering

下载好依赖的python module以及Adult数据集，运行clustering analyse.py即可复现实验

### 1. 数据预处理
数据集中包含14个特征，包括age、education-num、occupation、hours-per-week等信息，以及一个分类标准income。对数据进行如下预处理：

·删除education列，因为education列为字符串，而且与education-num，一个纯数字，高度关联，因此我认为education-num已经能够表达education所表达的信息，可以将education列删除

·缺失值处理：将?替换为NaN并删除

·获取分类标签：将income为”>50K”的数据标记为1，”<=50K”的数据标记为0，并在原表格中删除income这一列
  
·对非数值变量进行独热编码

·对数值型变量进行标准化

### 2. 算法实现以及运行

①对于传统聚类算法BIRCH，K-means，Agglomerative Clustering，直接调用sklearn库里的函数即可，并调用adjusted_rand_score, normalized_mutual_info_score函数分别计算ARI和NMI值

②对于深度学习算法，我设计了一个简易的AutoEncoder+K-means模型，使用自编码器对初始数据进行编码以进行特征提取，再交给K-means算法进行聚类

③最后通过PCA对数据进行降维，并将聚类结果可视化

## 分类实验：

如果有小伙伴想要分享回归任务的实验，请联系我

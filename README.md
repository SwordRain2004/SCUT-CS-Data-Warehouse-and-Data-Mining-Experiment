# SCUT-CS-Data-Warehouse-and-Data-Mining-Experiment
华南理工大学计科全英创新班2025年春数据仓库与数据挖掘实验  --更新中

实验前置任务算法与数据集的确定：使用中文 Word Embedding 选取与自己姓名最	接近的(使用2-范数)三个算法以及数据集

（1）运行get uci database names.py文件，获取uci数据库的中英文名称，在代码中使用了爬虫技术以及百度翻译的api

（2）运行calculate embedding vector.py文件，计算姓名、算法中文名、数据库中文名的向量，在代码中使用了阿里云百炼的text-embedding-v3模型的api将中文文本向量化

（3）运行calculate enclidean distance.py文件，计算欧几里得距离（即2-范数），确定所需要实现的算法和使用的数据库

作者的任务是完成聚类部分的实验，目前还没有完成

如果有小伙伴想要分享回归任务的实验，请联系我

"""本代码通过百度翻译api和爬虫技术获取uci数据库的英文名和中文翻译并保存在uci_datasets.xlsx中"""


import requests
from bs4 import BeautifulSoup
import pandas as pd
import time


def baidufanyi(query='None',  # 被翻译的文本
               from_lang='en',  # 被翻译的文本的语言
               to_lang='zh',  # 目标语言
               appid='2025xxxxxxxxxxxxx',  # 百度翻译开发者中心APP ID
               appkey='_Swxxxxxxxxxxxxxxxxx'  # 百度翻译开发者中心密钥
               ):
    """百度翻译API申请：https://fanyi-api.baidu.com/"""
    import random
    import json
    from hashlib import md5
    def make_md5(s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)
    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()
    return result['trans_result']


def uci():
    headers = {
        "User-Agent": ""
    }  # 更换为自己的User-Agent，如果不知道自己的User-Agent是什么，请自行上网搜索获取方法
    Dataset = []
    for start_num in range(0, 678, 25):
        response = requests.get(
            f"http://archive.ics.uci.edu/datasets/?skip={start_num}&take=25&sort=desc&orderBy=NumHits&search=",
            headers=headers)
        temp = []
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        all_titles = soup.findAll("a", attrs={"class": ["link-hover link text-xl font-semibold"]})
        q = ''
        for title in all_titles:
            CN_title = title.string
            q += CN_title + '\n'
            temp.append(CN_title)
        result = baidufanyi(query=q)
        trans_name = []
        for i in result:
            trans_name.append(i["dst"])
        for i in range(0, len(temp)):
            Dataset.append([temp[i], trans_name[i]])
    df = pd.DataFrame(Dataset, columns=["Dataset Name", "数据库名字"])
    df.to_excel("uci_datasets.xlsx", index=False)
uci()

import spacy
from wikipediaapi import Wikipedia
import requests
from bs4 import BeautifulSoup
import re
import json
from utils import *
from tqdm import tqdm

def get_result(response):
    regex = r"Q(\d+)"
    gold_id = None
    try:
        match = re.search(regex,response).group(1)
        gold_id = f"Q{match}"
    except:
        try:
            regex = r"NIR_(.*)"
            match = re.search(regex,response).group(1)
            gold_id = f"NIR_{match}"
        except: pass
    return gold_id

def link_entities(text, topk=5):
    # 初始化 spaCy 英文模型
    nlp = spacy.load("zh_core_web_sm")
    proxies = {
        
    }
    # 初始化 Wikipedia API
    headers ={"User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'}
    wiki_wiki = Wikipedia('en',headers=headers)

    # 使用 spaCy 进行实体识别
    doc = nlp(text)

    # 存储实体及其链接结果
    entities_linked = []
    ents = doc.ents
    if len(ents)>topk:
        ents = ents[:topk]
    for ent in ents:
        # 使用 Wikipedia API 查询实体
        page = wiki_wiki.page(ent.text)
        if page.exists():
            # 如果页面存在，则提供链接和摘要
            response = requests.get(page.fullurl,headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            try:
                wikidata_link = soup.find("a", href=lambda href: href and "wikidata.org" in href)
                wikidata_id = wikidata_link.get('href').split('/')[-1] if wikidata_link else None
                gold_id = get_result(wikidata_id)
            except:
                gold_id = None
                
            entities_linked.append({
                'entity': ent.text,
                'wiki_url': page.fullurl,
                'gold_id': gold_id,
                'summary': page.summary[:200]  # 取摘要的前200字符
            })
        else:
            # 如果没有对应的页面，记录未找到
            entities_linked.append({'entity': ent.text, 'message': 'No matching page found on Wikipedia'})

    return entities_linked

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rank_entities(entities:list,
                  query:str):
    candidates = [entity.get('entity')+"\n" + entity.get('summary') for entity in entities]
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([query] + list(candidates))

    # 计算余弦相似度
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    # 输出每个候选实体的相似度分数
    entity_scores = {entity_info[0]: {"score":entity_info[1],"idx":idx} 
                     for idx, entity_info in enumerate(zip(candidates, cosine_sim[0]))}
    
    sorted_entities = dict(sorted(entity_scores.items(), key=lambda item: item[1]["score"], 
                                  reverse=True))

    return sorted_entities

# 测试代码
# text = "Apple Inc. is an American multinational technology company headquartered in Cupertino, California."
# result = link_entities(text)
# print(result)


test_dataset = readjsonl("entity_linking_project/zero-shot.jsonl")
test_ids = readinfo("entity_linking_project/test_res_spacy.json")

test_dataset = test_dataset[len(test_ids):]
try:
    for idx, entity_info in tqdm(enumerate(test_dataset)):

        text = get_info_description_en(entity_info)
        linked_entities = link_entities(text)
        test_ids.append(linked_entities)
        if idx%100 ==0:
                writeinfo("entity_linking_project/test_res_spacy.json",test_ids)
except:
    pass
writeinfo("entity_linking_project/test_res_spacy.json",test_ids)
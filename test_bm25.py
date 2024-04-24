import spacy
from wikipediaapi import Wikipedia
import requests
from bs4 import BeautifulSoup
import re
import json
from utils import *
from tqdm import tqdm

class CorpusParser:

	def __init__(self, filename):
		self.filename = filename
		self.regex = re.compile('^#\s*\d+')
		self.corpus = dict()

	def parse(self):
		with open(self.filename) as f:
			s = ''.join(f.readlines())
		blobs = s.split('#')[1:]
		for x in blobs:
			text = x.split()
			docid = text.pop(0)
			self.corpus[docid] = text

	def get_corpus(self):
		return self.corpus


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


def test_get_corpus():
    from BM25.src.parse import CorpusParser
    

def link_entities(text, topk=5):
    # 初始化 spaCy 英文模型
    nlp = spacy.load("zh_core_web_sm")
    
    
    # 使用 spaCy 进行实体识别
    doc = nlp(text)

    # 存储实体及其链接结果
    entities_linked = []
    ents = doc.ents
    
    
    
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
            entities_linked.append({'entity': ent.text, 'gold_id': None})
    entities_linked = get_sort_entities(entities_linked,text)
    return entities_linked


def get_sort_entities(entities,query):
    entities = list(filter(lambda x:x["gold_id"] is not None, entities))
    if len(entities) ==0:
        return {}
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # 示例数据
    text = query
    candidates = {entity['gold_id']: entity['summary'] for entity in entities}
    # 创建 TF-IDF 模型
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([text] + list(candidates.values()))

    # 计算余弦相似度
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    # 输出每个候选实体的相似度分数
    entity_scores = {entity: score for entity, score in zip(candidates, cosine_sim[0])}
    sorted_entities = dict(sorted(entity_scores.items(), key=lambda item: item[1], reverse=True))

    return sorted_entities

# 测试代码
# text = "Apple Inc. is an American multinational technology company headquartered in Cupertino, California."
# result = link_entities(text)
# print(result)


test_dataset = readjsonl("entity_linking_project/zero-shot.jsonl")
test_ids = readinfo("entity_linking_project/test_res_bm25.json")

test_dataset = test_dataset[len(test_ids):]
try:
    for idx, entity_info in tqdm(enumerate(test_dataset)):

        text = get_info_description_en(entity_info)
        linked_entities = link_entities(text)
        test_ids.append(linked_entities)
        if idx%100 ==0:
                writeinfo("entity_linking_project/test_res_bm25.json",test_ids)
except:
    pass
writeinfo("entity_linking_project/test_res_bm25.json",test_ids)
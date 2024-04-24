import spacy
from wikipediaapi import Wikipedia

def link_entities(text):
    # 初始化 spaCy 英文模型
    nlp = spacy.load("en_core_web_sm")
    # 初始化 Wikipedia API
    headers ={"User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'}
    wiki_wiki = Wikipedia('en',headers=headers)

    # 使用 spaCy 进行实体识别
    doc = nlp(text)

    # 存储实体及其链接结果
    entities_linked = []

    for ent in doc.ents:
        # 使用 Wikipedia API 查询实体
        page = wiki_wiki.page(ent.text)
        if page.exists():
            # 如果页面存在，则提供链接和摘要
            wikidata_id = page.data['pageprops'].get('wikibase_item',"null")
            entities_linked.append({
                'entity': ent.text,
                'wiki_url': page.fullurl,
                'summary': page.summary[:200],  # 取摘要的前200字符
                'gold_id': wikidata_id
            })
        else:
            # 如果没有对应的页面，记录未找到
            entities_linked.append({'entity': ent.text, 'message': 'No matching page found on Wikipedia'})

    return entities_linked

# 测试代码
text = "Apple Inc. is an American multinational technology company headquartered in Cupertino, California."
result = link_entities(text)
print(result)

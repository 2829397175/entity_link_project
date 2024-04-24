from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from utils import *


def test_auc(y_true,
             y):
    # 计算AUC
    pred=0
    for y_t,y_p in zip(y_true,y):
       if y_t == y_p:
           pred+=1
    print('AUC:{auc:.2f}'.format(auc =pred/len(y_true)))
    
if __name__ =="__main__":
    y_true = readjsonl("entity_linking_project/zero-shot.jsonl")
    
    ## spacy
    y_test = readinfo("entity_linking_project/test_res_spacy.json")
    
   
    y_true = [x["gold_id"] for x in y_true]
    y_test_filtered = []
    for y_p in y_test:
        y_p_  = list(y_p.keys())
        y_test_filtered.append(
           y_p_[0] if len(y_p_)>1 else None
        )
    
    # ## gpt-4
    # y_test_filtered = readinfo("entity_linking_project/test_res_gpt-4-turbo.json")
    
    ## gpt-3.5
    y_test_filtered = readinfo("entity_linking_project/test_res_gpt-3.5-turbo.json")
    
    tested_len = len(y_test_filtered)
    if tested_len< len(y_true):
        y_true = y_true[:tested_len]
    test_auc(y_true,y_test_filtered)
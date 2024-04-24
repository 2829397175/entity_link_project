import json
import os

def get_info_description(info):
    filtered_ks =["id","gold_id","start","end"]

    start = info['start']
    end = info['end']
    info['text'] = info['text'][:start] + '**' + info['text'][start:end] + '**' + info['text'][end:]

    k_dict = {
        'text': '上下文',
        'mention': '实体',
        'source': '来源',
        'domain': '领域'
    }

    infos = [f"{k_dict[k]}:{v}" if k not in filtered_ks else "" for k,v in info.items()]
    infos = [kv for kv in infos if kv]

    return "\n".join(infos)


def get_info_description_en(info):
    filtered_ks =["id","gold_id","start","end"]
    infos = [f"{k}:{v}" if k not in filtered_ks else "" for k,v in info.items() ]
    return "\n".join(infos)


def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list


def writeinfo(data_dir,info):
    with open(data_dir,'w',encoding = 'utf-8') as f:
            json.dump(info, f, indent=4,separators=(',', ':'),ensure_ascii=False)



def readjsonl(data_dir):
    data = []
    with open(data_dir, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data
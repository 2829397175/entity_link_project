import agentscope

agentscope.init(model_configs="llm_configs.json")

from agentscope.models import _MODEL_CONFIGS,load_model_by_config_name
from agentscope.agents import ReActAgent
import json
import os
import re

from agentscope.message import Msg
from tqdm import tqdm
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
    
def get_dataset_res(config_name,
                    prompt_model):
    test_dataset = readjsonl("zero-shot.jsonl")
    test_ids = []
    
    sys_msg = Msg("system", "You're a helpful assistant.", role="system")
    for entity_info in test_dataset:
        entity_info_nlp = get_info_description(entity_info)
        model = load_model_by_config_name(config_name)
        prompt_template = prompt_model["prompt_template"]
        input_variables = prompt_model["input_variables"]


        inputs = {
            k:entity_info_nlp for k in input_variables
        }

        prompt = prompt_template.format_map(inputs)

        print(prompt)
        gold_id = None
        prompts = [sys_msg,Msg("user",prompt,"user")]
        # prepare prompt
        prompt = model.format(
            *prompts
        )


        response = model(prompt)
        response = response.text

        print(response)
        regex = r"Q(\d+)"
        
        try:
            match = re.search(regex,response).group(1)
            gold_id = f"Q{match}"
        except:
            try:
                regex = r"NIR_(.*)"
                match = re.search(regex,response).group(1)
                gold_id = f"NIR_{match}"
            except: pass
        
        test_ids.append(gold_id) 
        
        
        # agent = ReActAgent(
        # name="assistant",
        # model_config_name=config_name,
        # tools=[],
        # sys_prompt = prompt,
        # verbose=True, # set verbose to True to show the reasoning process
        # )
        # agent(entity_info_nlp)
    writeinfo("entity_linking_project/test_res.json",test_ids)
        
    
    
prompts = readinfo("prompts.json")

prompts_map = {}

for prompt in prompts:
    prompts_map[prompt["config_name"]] = prompt

for config_name in _MODEL_CONFIGS.keys():
    
    
    get_dataset_res(config_name,
                    prompts_map[config_name])

# for config_name in 
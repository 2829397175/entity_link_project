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
    with open(data_dir, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def get_info_description(info):
    filtered_ks =["id","gold_id","start","end"]
    infos = [f"{k}:{v}" if k not in filtered_ks else "" for k,v in info.items() ]
    return "\n".join(infos)
    
def get_dataset_res(config_name,
                    prompt_model):
    test_dataset = readjsonl("entity_linking_project/zero-shot.jsonl")
    try:
        test_ids = readinfo(f"entity_linking_project/test_res_{config_name}.json")
    except:
        test_ids =[]
    tested_len = len(test_ids)
    test_dataset = test_dataset[tested_len:]
    
    sys_msg = Msg("system", "You're a helpful assistant.", role="system")
    model = load_model_by_config_name(config_name)
    for idx,entity_info in tqdm(enumerate(test_dataset)):
        entity_info_nlp = get_info_description(entity_info)
        prompt_template = prompt_model["prompt_template"]
        input_variables = prompt_model["input_variables"]
        inputs ={
            k:entity_info_nlp for k in input_variables
        }
        
        prompt = prompt_template.format_map(inputs)
        gold_id = None
        prompts =[sys_msg,Msg("user",prompt,"user")]
        # prepare prompt
        prompt = model.format(
            *prompts
        )
        response = model(prompt)
        response = response.text
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
        # except Exception as e:
        #     pass
        
        test_ids.append(gold_id) 
        if idx%100 ==0:
            writeinfo(f"entity_linking_project/test_res_{config_name}.json",test_ids)
        
        # agent = ReActAgent(
        # name="assistant",
        # model_config_name=config_name,
        # tools=[],
        # sys_prompt = prompt,
        # verbose=True, # set verbose to True to show the reasoning process
        # )
        # agent(entity_info_nlp)
    writeinfo(f"entity_linking_project/test_res_{config_name}.json",test_ids)
        
    
    
prompts = readinfo("prompts.json")

prompts_map = {}

for prompt in prompts:
    prompts_map[prompt["config_name"]] = prompt

config_keys = list(_MODEL_CONFIGS.keys())
config_keys = ["gpt-4-turbo"]

for config_name in config_keys:
    get_dataset_res(config_name,
                    prompts_map[config_name])

# for config_name in 
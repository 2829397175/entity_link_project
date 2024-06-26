import agentscope

agentscope.init(model_configs="llm_configs.json")

from agentscope.models import _MODEL_CONFIGS,load_model_by_config_name
from agentscope.agents import ReActAgent
import json
import os
import re

from agentscope.message import Msg
from tqdm import tqdm
from utils import *


    
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
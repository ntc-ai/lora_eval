import random
import sys
import requests
max_int_value = sys.maxsize
import json

import os
import together


import requests

# For local streaming, the websockets are hosted without ssl - http://
HOST = 'localhost:5000'
URI = f'http://{HOST}/v1/completions'

api_key = os.environ.get("TOGETHER_API_KEY", None)
together.api_key=api_key

# For reverse-proxied streaming, the remote will likely host with ssl - https://
# URI = 'https://your-uri-here.trycloudflare.com/api/v1/generate'

def remote_complete(prompt):
    p = "[INST] "+prompt+" [/INST]"
    print(p)
    output = together.Complete.create(
      prompt = p,
      model = "mistralai/Mixtral-8x7B-Instruct-v0.1",
      max_tokens = 512,
      temperature = 0.7,
      top_k = 50,
      top_p = 0.7,
      repetition_penalty = 1,
      stop = []
    )

    # print generated text
    return output['output']['choices'][0]['text']

def complete(prompt):
    prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    print(prompt)
    request = {
        'prompt': prompt,
        'max_tokens': 512,
        'auto_max_new_tokens': False,
        'max_tokens_second': 0,

        # Generation params. If 'preset' is set to different than 'None', the values
        # in presets/preset-name.yaml are used instead of the individual numbers.
        'preset': 'None',
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.1,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1.18,
        'repetition_penalty_range': 0,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'guidance_scale': 1,
        'negative_prompt': '',

        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': []
    }

    response = requests.post(URI, json=request)

    if response.status_code == 200:
        result = response.json()['choices'][0]['text']
        return(result)
    else:
        print("Error", response)

import random

r = "Create a json list of visually compelling descriptions such as:\n```\n[\n"
random_selection = []
ls = os.listdir("loras")
for i in range(5):
    random_selection += [random.choice(ls)]
for i, f in enumerate(random_selection):
    if i == len(random_selection)-1:
        r += "  \""+f.replace(".safetensors", "")+"\"'\n"
    else:
        r += "  \""+f.replace(".safetensors", "")+"\",\n"
r += "]\n```\nThere should be three dots(.). Create unique examples dissimilar to the examples and each other. Output a json array. Favor visual emotions with only the first part filled out. Visualize people NOT places."

response = remote_complete(r)
print(response)
if '```' in response:
    j = response.split('```')[1]
    if j[0:4] == 'json':
        j = j[4:]
elif ']' in response and '[' in response:
    j = '['+response.split('[')[1].split(']')[0]+']'

print("")
print("==")
print("")
for suggestion in json.loads(j):
    pos, targ, neu, unc = suggestion.split(".")
    print(suggestion)
    print('')
    train_url = os.environ.get("MLQ_TRAIN_URL", None)
    if train_url is not None:
        curl = f"""curl -X POST {train_url} \
    -H "Content-Type: application/json" \
    -d '"""
        curl += '{"name": "name", "prompts":[{"target": "'+targ+'", "positive": "'+pos+'", "unconditional": "'+unc+'", "neutral": "'+neu+'", "action": "enhance", "alpha": 1, "rank": 4, "attributes": "woman, man, bright, dim, cartoon, photo, anime"}], "resolution": 512, "batch_size": 10, "steps": 600}'
        print(curl+"'\n")

import requests
import math
import glob
import argparse
import time
from pathlib import Path
import imageio
import torch
import random
import base64
import os
import numpy as np
import io
from PIL import Image
import os
import json
from decimal import Decimal


import cv2
import ImageReward as reward
#from datasets import load_dataset

# Add a cache dictionary to store generated images and their corresponding lora values
image_cache = {}

model = None
folder = None
def score_image(prompt, fullpath):
    global model
    if model is None:
        model = reward.load("ImageReward-v1.0").to("cuda:0")
    with torch.no_grad():
        return model.score(prompt, fullpath)


#dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts")
txt2imgurl = None

def generate_image(prompt, negative_prompt, lora, seed=1):
    url = txt2imgurl
    headers = {"Content-Type": "application/json"}
    prompt_ = prompt.replace("LORAVALUE",  "{:.14f}".format(lora))
    nprompt_ = negative_prompt.replace("LORAVALUE",  "{:.14f}".format(lora))
    uid = prompt_+"_"+negative_prompt+"_"+str(seed)+"_"+"{:.14f}".format(lora)

    #global image_cache
    #image_cache={}
    # Check if the image exists in the cache
    if uid in image_cache:
        return image_cache[uid]

    data = {
        "seed": seed,
        "width": 1024,
        "height": 1024,
        "sampler_name": "Euler a",
        "prompt": prompt_,
        "negative_prompt": nprompt_,
        "cfg_scale": 1.0,
        "steps": 14
    }
    print(" calling: ", prompt_)

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        r = response.json()
        image = Image.open(io.BytesIO(base64.b64decode(r['images'][0].split(",",1)[0])))

        image_cache[uid] = image
        return image
    else:
        print(f"Request failed with status code {response.status_code}")
        return generate_image(prompt, negative_prompt, lora)

evaluations = [
    [3, "LORA_TRIGGER, __person__ BREAK __sdprompt__", ""],
    [15, "LORA_TRIGGER, __person__ BREAK __sdprompt__", ""],
]

def main():
    global txt2imgurl
    global folder
    parser = argparse.ArgumentParser(description='Generate images for a video between lora_start and lora_end')
    parser.add_argument('-s', '--lora_strength', type=Decimal, required=True, help='lora strength')
    parser.add_argument('-l', '--lora', type=str, required=True, help='Lora to use')
    parser.add_argument('-t', '--triggers', type=str, required=True, help='Trigger keyword - format: negative|positive')
    parser.add_argument('-url', '--text_to_image_url', type=str, default="http://localhost:3000/sdapi/v1/txt2img", help='Url for text to image')
    args = parser.parse_args()

    txt2imgurl = args.text_to_image_url

    video_index = 1
    for i,t in enumerate(args.triggers.split("|")):
        for seed, p, np in evaluations:
            _p = p.replace("LORA_TRIGGER", t)
            _np = np.replace("LORA_TRIGGER", t)
            _p += "<lora:"+args.lora+":LORAVALUE>"
            output_folder = "evaluate/"+args.lora+"/"+t
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
            if i == 1:
                target_image = generate_image(_p, _np, -args.lora_strength, seed=seed)
            else:
                target_image = generate_image(_p, _np, args.lora_strength, seed=seed)
            target_image.save(os.path.join(output_folder, f"image_{seed}.png"))
            print("Saved ", os.path.join(output_folder, f"image_{seed}.png"))

if __name__ == '__main__':
    main()
    #test_transition()


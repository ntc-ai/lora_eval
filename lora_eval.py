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
#import ImageReward as reward
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

def generate_image(txt2imgurl, prompt, negative_prompt, lora, seed=1):
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

    response = requests.post(txt2imgurl, headers=headers, data=json.dumps(data))
 
    if response.status_code == 200:
        r = response.json()
        image = Image.open(io.BytesIO(base64.b64decode(r['images'][0].split(",",1)[0])))

        image_cache[uid] = image
        return image
    else:
        print(f"Request failed with status code {response.status_code}")
        return generate_image(txt2imgurl, prompt, negative_prompt, lora)

evaluations = json.loads(open("evaluations.json","r").read())

def lora_eval(txt2imgurl, lora, triggers, lora_strengths):
    output_folder = "evaluate/"+lora
    for t in triggers:
        if t == "":
            continue
        for seed, p, np in evaluations:
            seedg = random.SystemRandom().randint(0, 2**32-1)
            for lora_strength in lora_strengths:
                lora_strength = float(lora_strength)
                _p = p.replace("LORA_TRIGGER", t)
                _np = np.replace("LORA_TRIGGER", t)
                _p = _p.replace("INSERT_LORA", "<lora:"+lora+":LORAVALUE>")
                _np = _np.replace("INSERT_LORA", "<lora:"+lora+":LORAVALUE>")
                output_file = os.path.join(output_folder, "{:s}_{:d}_{:.1f}.png".format(t,seed,lora_strength))
                if os.path.exists(output_file):
                    #print("Skipping", output_file, "(exists)")
                    continue
                output_path = Path(output_folder)
                output_path.mkdir(parents=True, exist_ok=True)
                target_image = generate_image(txt2imgurl, _p, _np, lora_strength, seed=seedg)
                target_image.save(output_file)
                print("Saved ", os.path.join(output_folder, f"image_{seed}.png"))


def main():
    parser = argparse.ArgumentParser(description='Generate images for evaluating a lora')
    parser.add_argument('-s', '--lora_strength', nargs='*', required=True, help='lora strength, can be multiple: -2 0 2')
    parser.add_argument('-l', '--lora', type=str, required=True, help='Lora to use')
    parser.add_argument('-t', '--triggers', nargs='*', required=True, help='Trigger keyword - can pass in multiple: happy sad')
    parser.add_argument('-url', '--text_to_image_url', type=str, default="http://localhost:3000/sdapi/v1/txt2img", help='Url for text to image')
    args = parser.parse_args()

    txt2imgurl = args.text_to_image_url
    lora_eval(txt2imgurl, args.lora, args.triggers, args.lora_strength)

if __name__ == '__main__':
    main()
    #test_transition()


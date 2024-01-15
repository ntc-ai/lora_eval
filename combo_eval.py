import os
import time
import re
import json
import random
import requests
import sys
import shutil
import argparse

import uuid

# Path to the directory
directory = 'loras/'

# List all files in the directory
all_files = os.listdir(directory)

# Filter files with four dots in the filename and ending with .safetensors
filtered_files = [file for file in all_files if file.count('.') == 4 and file.endswith('.safetensors')]

file_cache = {}

def load_file_and_return_random_line(file_path):
    global file_cache

    # If the file content is not in cache, load it
    if file_path not in file_cache:
        with open(file_path, 'r') as file:
            file_cache[file_path] = file.read().split('\n')

    # If the file content is empty or only contains '', return None
    if not file_cache[file_path] or file_cache[file_path] == ['']:
        return None

    # Return a random line
    line = random.choice(file_cache[file_path])

    return line

def wildcard_replace(s, directory):
    if directory is None:
        return s
    # Use a regular expression to find all occurrences of '__...__' in the string
    wildcards = re.findall(r'__(.*?)__', s)

    # Load a random line from the file corresponding to each wildcard
    replaced = [load_file_and_return_random_line(directory+"/"+w+".txt") for w in wildcards]

    # Create a dictionary mapping each wildcard to its replacement
    replacements = dict(zip(wildcards, replaced))

    # Replace each occurrence of '__...__' in the string with its corresponding replacement
    for wildcard, replacement in replacements.items():
        s = s.replace('__{}__'.format(wildcard), replacement)

    return s

def sample_seed(loras, seed, prompt, uuid, savedir, model, lambda_val, p_val, steps, cfg, scale, strength):
    lorain = {
            'input_mode': 'simple',
            'lora_count': len(loras),
            }
    lorainlist=[]
    loralength = 0
    for i in range(49):
        lorain['lora_wt_'+str(i+1)]=1
        lorain['model_str_'+str(i+1)]=1
        lorain['clip_str_'+str(i+1)]=1
        if i < len(loras):
            lorain['lora_name_'+str(i+1)]="Lora/sdxl/sliders/mlq/"+loras[i]
            lorainlist += ["Lora/ntc-sdxl-sliders/"+loras[i], 1, 1, 1]
            loralength+=1
        else:
            lorain['lora_name_'+str(i+1)]="None"
            lorainlist += ["None", 1, 1, 1]

    headers = {
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'Origin': 'http://192.168.0.180:8189',
            'Pragma': 'no-cache',
            'Referer': 'http://192.168.0.180:8189/',
            'Sec-GPC': '1',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
            }
    json_data = {
        'client_id': '702201dd6fd949c686ab4223167036e7',
        'prompt': {
            '5': {
                'inputs': {
                    'width': 896,
                    'height': 1152,
                    'batch_size': 1,
                },
                'class_type': 'EmptyLatentImage',
            },
            '7': {
                'inputs': {
                    'text': prompt[2],
                    'clip': [
                        '55',
                        1,
                    ],
                },
                'class_type': 'CLIPTextEncode',
            },
            '8': {
                'inputs': {
                    'samples': [
                        '13',
                        0,
                    ],
                    'vae': [
                        '41',
                        2,
                    ],
                },
                'class_type': 'VAEDecode',
            },
            '13': {
                'inputs': {
                    'add_noise': True,
                    'noise_seed': [
                        '34',
                        0,
                    ],
                    'cfg': cfg,
                    'model': [
                        '55',
                        0,
                    ],
                    'positive': [
                        '57',
                        0,
                    ],
                    'negative': [
                        '7',
                        0,
                    ],
                    'sampler': [
                        '14',
                        0,
                    ],
                    'sigmas': [
                        '37',
                        0,
                    ],
                    'latent_image': [
                        '5',
                        0,
                    ],
                },
                'class_type': 'SamplerCustom',
            },
            '14': {
                'inputs': {
                    'sampler_name': 'euler_ancestral',
                },
                'class_type': 'KSamplerSelect',
            },
            '27': {
                'inputs': {
                    'filename_prefix': savedir+'/'+str(uuid),
                    'images': [
                        '8',
                        0,
                    ],
                },
                'class_type': 'SaveImage',
            },
            '34': {
                'inputs': {
                    'seed': seed,
                },
                'class_type': 'CR Seed',
            },
            '37': {
                'inputs': {
                    'scheduler': 'simple',
                    'steps': steps,
                    'denoise': 1.0,
                    'model': [
                        '55',
                        0,
                    ],
                },
                'class_type': 'BasicScheduler',
            },
            '41': {
                'inputs': {
                    'ckpt_name': model,
                },
                'class_type': 'CheckpointLoaderSimple',
            },
            '48': {
                'inputs': {
                    'filename_prefix': 'LoRAs/'+uuid,
                    'LoRA': [
                        '54',
                        0,
                    ],
                },
                'class_type': 'Save LoRA',
            },
            '49': {
                'inputs': {
                    'text': prompt[1],
                    'clip': [
                        '55',
                        1,
                    ],
                },
                'class_type': 'CLIPTextEncode',
            },
            '50': {
                'inputs': lorain,
                'class_type': 'LoRA Stacker',
            },
            '51': {
                'inputs': {
                    'Value': strength,
                },
                'class_type': 'Float',
            },
            '54': {
                'inputs': {
                    'lambda_val': lambda_val,
                    'p': p_val,
                    'scale': scale,
                    'seed': 1,
                    'lora_stack': [
                        '50',
                        0,
                    ],
                },
                'class_type': 'DARE Merge LoRA Stack',
            },
            '55': {
                'inputs': {
                    'lora_model_wt': [
                        '51',
                        0,
                    ],
                    'lora_clip_wt': [
                        '51',
                        0,
                    ],
                    'LoRA': [
                        '54',
                        0,
                    ],
                    'model': [
                        '41',
                        0,
                    ],
                    'clip': [
                        '41',
                        1,
                    ],
                },
                'class_type': 'Apply LoRA',
            },
            '56': {
                'inputs': {
                    'text': prompt[0],
                    'clip': [
                        '55',
                        1,
                    ],
                },
                'class_type': 'CLIPTextEncode',
            },
            '57': {
                'inputs': {
                    'conditioning_to': [
                        '56',
                        0,
                    ],
                    'conditioning_from': [
                        '49',
                        0,
                    ],
                },
                'class_type': 'ConditioningConcat',
            },
        },
        'extra_data': {
            'extra_pnginfo': {
                'workflow': {
                    'last_node_id': 69,
                    'last_link_id': 196,
                    'nodes': [
                        {
                            'id': 27,
                            'type': 'SaveImage',
                            'pos': [
                                1211.7722083984381,
                                -44.8865373046875,
                            ],
                            'size': {
                                '0': 466.7873840332031,
                                '1': 516.8289794921875,
                            },
                            'flags': {},
                            'order': 16,
                            'mode': 0,
                            'inputs': [
                                {
                                    'name': 'images',
                                    'type': 'IMAGE',
                                    'link': 54,
                                },
                            ],
                            'properties': {},
                            'widgets_values': [
                                savedir+'_/'+uuid,
                            ],
                            'color': '#322',
                            'bgcolor': '#533',
                        },
                        {
                            'id': 8,
                            'type': 'VAEDecode',
                            'pos': [
                                785,
                                266,
                            ],
                            'size': {
                                '0': 210,
                                '1': 46,
                            },
                            'flags': {},
                            'order': 15,
                            'mode': 0,
                            'inputs': [
                                {
                                    'name': 'samples',
                                    'type': 'LATENT',
                                    'link': 28,
                                },
                                {
                                    'name': 'vae',
                                    'type': 'VAE',
                                    'link': 106,
                                    'slot_index': 1,
                                },
                            ],
                            'outputs': [
                                {
                                    'name': 'IMAGE',
                                    'type': 'IMAGE',
                                    'links': [
                                        54,
                                    ],
                                    'slot_index': 0,
                                },
                            ],
                            'properties': {
                                'Node name for S&R': 'VAEDecode',
                                'ttNbgOverride': {
                                    'color': '#432',
                                    'bgcolor': '#653',
                                    'groupcolor': '#b58b2a',
                                },
                            },
                            'color': '#432',
                            'bgcolor': '#653',
                        },
                        {
                            'id': 13,
                            'type': 'SamplerCustom',
                            'pos': [
                                331,
                                -40,
                            ],
                            'size': {
                                '0': 355.20001220703125,
                                '1': 442,
                            },
                            'flags': {},
                            'order': 14,
                            'mode': 0,
                            'inputs': [
                                {
                                    'name': 'model',
                                    'type': 'MODEL',
                                    'link': 186,
                                    'slot_index': 0,
                                },
                                {
                                    'name': 'positive',
                                    'type': 'CONDITIONING',
                                    'link': 166,
                                    'slot_index': 1,
                                },
                                {
                                    'name': 'negative',
                                    'type': 'CONDITIONING',
                                    'link': 20,
                                },
                                {
                                    'name': 'sampler',
                                    'type': 'SAMPLER',
                                    'link': 62,
                                    'slot_index': 3,
                                },
                                {
                                    'name': 'sigmas',
                                    'type': 'SIGMAS',
                                    'link': 82,
                                    'slot_index': 4,
                                },
                                {
                                    'name': 'latent_image',
                                    'type': 'LATENT',
                                    'link': 23,
                                    'slot_index': 5,
                                },
                                {
                                    'name': 'noise_seed',
                                    'type': 'INT',
                                    'link': 68,
                                    'widget': {
                                        'name': 'noise_seed',
                                    },
                                },
                            ],
                            'outputs': [
                                {
                                    'name': 'output',
                                    'type': 'LATENT',
                                    'links': [
                                        28,
                                    ],
                                    'shape': 3,
                                    'slot_index': 0,
                                },
                                {
                                    'name': 'denoised_output',
                                    'type': 'LATENT',
                                    'links': None,
                                    'shape': 3,
                                },
                            ],
                            'properties': {
                                'Node name for S&R': 'SamplerCustom',
                                'ttNbgOverride': {
                                    'color': '#432',
                                    'bgcolor': '#653',
                                    'groupcolor': '#b58b2a',
                                },
                            },
                            'widgets_values': [
                                True,
                                156,
                                'fixed',
                                cfg,
                            ],
                            'color': '#432',
                            'bgcolor': '#653',
                        },
                        {
                            'id': 14,
                            'type': 'KSamplerSelect',
                            'pos': [
                                -105,
                                -32,
                            ],
                            'size': {
                                '0': 315,
                                '1': 58,
                            },
                            'flags': {},
                            'order': 0,
                            'mode': 0,
                            'outputs': [
                                {
                                    'name': 'SAMPLER',
                                    'type': 'SAMPLER',
                                    'links': [
                                        62,
                                    ],
                                    'shape': 3,
                                    'slot_index': 0,
                                },
                            ],
                            'properties': {
                                'Node name for S&R': 'KSamplerSelect',
                                'ttNbgOverride': {
                                    'color': '#432',
                                    'bgcolor': '#653',
                                    'groupcolor': '#b58b2a',
                                },
                            },
                            'widgets_values': [
                                'euler_ancestral',
                            ],
                            'color': '#432',
                            'bgcolor': '#653',
                        },
                        {
                            'id': 37,
                            'type': 'BasicScheduler',
                            'pos': [
                                -99,
                                123,
                            ],
                            'size': {
                                '0': 315,
                                '1': 82,
                            },
                            'flags': {},
                            'order': 9,
                            'mode': 0,
                            'inputs': [
                                {
                                    'name': 'model',
                                    'type': 'MODEL',
                                    'link': 185,
                                },
                            ],
                            'outputs': [
                                {
                                    'name': 'SIGMAS',
                                    'type': 'SIGMAS',
                                    'links': [
                                        82,
                                    ],
                                    'shape': 3,
                                    'slot_index': 0,
                                },
                            ],
                            'properties': {
                                'Node name for S&R': 'BasicScheduler',
                                'ttNbgOverride': {
                                    'color': '#432',
                                    'bgcolor': '#653',
                                    'groupcolor': '#b58b2a',
                                },
                            },
                            'widgets_values': [
                                'simple',
                                12
                            ],
                            'color': '#432',
                            'bgcolor': '#653',
                        },
                        {
                            'id': 5,
                            'type': 'EmptyLatentImage',
                            'pos': [
                                -98,
                                388,
                            ],
                            'size': {
                                '0': 315,
                                '1': 106,
                            },
                            'flags': {},
                            'order': 1,
                            'mode': 0,
                            'outputs': [
                                {
                                    'name': 'LATENT',
                                    'type': 'LATENT',
                                    'links': [
                                        23,
                                    ],
                                    'slot_index': 0,
                                },
                            ],
                            'properties': {
                                'Node name for S&R': 'EmptyLatentImage',
                                'ttNbgOverride': {
                                    'color': '#432',
                                    'bgcolor': '#653',
                                    'groupcolor': '#b58b2a',
                                },
                            },
                            'widgets_values': [
                                896,
                                1152,
                                1,
                            ],
                            'color': '#432',
                            'bgcolor': '#653',
                        },
                        {
                            'id': 49,
                            'type': 'CLIPTextEncode',
                            'pos': [
                                -1707,
                                -231,
                            ],
                            'size': {
                                '0': 425.27801513671875,
                                '1': 180.6060791015625,
                            },
                            'flags': {},
                            'order': 11,
                            'mode': 0,
                            'inputs': [
                                {
                                    'name': 'clip',
                                    'type': 'CLIP',
                                    'link': 189,
                                },
                            ],
                            'outputs': [
                                {
                                    'name': 'CONDITIONING',
                                    'type': 'CONDITIONING',
                                    'links': [
                                        148,
                                    ],
                                    'slot_index': 0,
                                },
                            ],
                            'title': 'Positive Prompt',
                            'properties': {
                                'Node name for S&R': 'Positive1',
                            },
                            'widgets_values': [
                                prompt[1],
                            ],
                            'color': '#232',
                            'bgcolor': '#353',
                        },
                        {
                            'id': 56,
                            'type': 'CLIPTextEncode',
                            'pos': [
                                -1705,
                                -475,
                            ],
                            'size': {
                                '0': 425.27801513671875,
                                '1': 180.6060791015625,
                            },
                            'flags': {},
                            'order': 10,
                            'mode': 0,
                            'inputs': [
                                {
                                    'name': 'clip',
                                    'type': 'CLIP',
                                    'link': 188,
                                },
                            ],
                            'outputs': [
                                {
                                    'name': 'CONDITIONING',
                                    'type': 'CONDITIONING',
                                    'links': [
                                        147,
                                    ],
                                    'slot_index': 0,
                                },
                            ],
                            'title': 'Positive Prompt',
                            'properties': {
                                'Node name for S&R': 'Positive0',
                            },
                            'widgets_values': [
                                prompt[0]
                            ],
                            'color': '#232',
                            'bgcolor': '#353',
                        },
                        {
                            'id': 57,
                            'type': 'ConditioningConcat',
                            'pos': [
                                -837,
                                -395,
                            ],
                            'size': {
                                '0': 380.4000244140625,
                                '1': 46,
                            },
                            'flags': {},
                            'order': 13,
                            'mode': 0,
                            'inputs': [
                                {
                                    'name': 'conditioning_to',
                                    'type': 'CONDITIONING',
                                    'link': 147,
                                },
                                {
                                    'name': 'conditioning_from',
                                    'type': 'CONDITIONING',
                                    'link': 148,
                                },
                            ],
                            'outputs': [
                                {
                                    'name': 'CONDITIONING',
                                    'type': 'CONDITIONING',
                                    'links': [
                                        166,
                                    ],
                                    'shape': 3,
                                    'slot_index': 0,
                                },
                            ],
                            'properties': {
                                'Node name for S&R': 'ConditioningConcat',
                                'ttNbgOverride': {
                                    'color': '#432',
                                    'bgcolor': '#653',
                                    'groupcolor': '#b58b2a',
                                },
                            },
                            'color': '#432',
                            'bgcolor': '#653',
                        },
                        {
                            'id': 7,
                            'type': 'CLIPTextEncode',
                            'pos': [
                                -1695,
                                1,
                            ],
                            'size': {
                                '0': 425.27801513671875,
                                '1': 180.6060791015625,
                            },
                            'flags': {},
                            'order': 12,
                            'mode': 0,
                            'inputs': [
                                {
                                    'name': 'clip',
                                    'type': 'CLIP',
                                    'link': 190,
                                },
                            ],
                            'outputs': [
                                {
                                    'name': 'CONDITIONING',
                                    'type': 'CONDITIONING',
                                    'links': [
                                        20,
                                    ],
                                    'slot_index': 0,
                                },
                            ],
                            'title': 'Negative Prompt',
                            'properties': {
                                'Node name for S&R': 'Negative',
                            },
                            'widgets_values': [
                                prompt[2],
                            ],
                            'color': '#232',
                            'bgcolor': '#353',
                        },
                        {
                            'id': 55,
                            'type': 'Apply LoRA',
                            'pos': [
                                -474,
                                153,
                            ],
                            'size': {
                                '0': 315,
                                '1': 122,
                            },
                            'flags': {},
                            'order': 8,
                            'mode': 0,
                            'inputs': [
                                {
                                    'name': 'LoRA',
                                    'type': 'LoRA',
                                    'link': 146,
                                },
                                {
                                    'name': 'model',
                                    'type': 'MODEL',
                                    'link': 193,
                                },
                                {
                                    'name': 'clip',
                                    'type': 'CLIP',
                                    'link': 194,
                                },
                                {
                                    'name': 'lora_model_wt',
                                    'type': 'FLOAT',
                                    'link': 143,
                                    'widget': {
                                        'name': 'lora_model_wt',
                                    },
                                },
                                {
                                    'name': 'lora_clip_wt',
                                    'type': 'FLOAT',
                                    'link': 144,
                                    'widget': {
                                        'name': 'lora_clip_wt',
                                    },
                                },
                            ],
                            'outputs': [
                                {
                                    'name': 'model',
                                    'type': 'MODEL',
                                    'links': [
                                        185,
                                        186,
                                    ],
                                    'shape': 3,
                                    'slot_index': 0,
                                },
                                {
                                    'name': 'clip',
                                    'type': 'CLIP',
                                    'links': [
                                        188,
                                        189,
                                        190,
                                    ],
                                    'shape': 3,
                                    'slot_index': 1,
                                },
                            ],
                            'properties': {
                                'Node name for S&R': 'Apply LoRA',
                            },
                            'widgets_values': [
                                1,
                                1,
                            ],
                            'color': '#223',
                            'bgcolor': '#335',
                        },
                        {
                            'id': 41,
                            'type': 'CheckpointLoaderSimple',
                            'pos': [
                                -1682,
                                483,
                            ],
                            'size': {
                                '0': 343.69647216796875,
                                '1': 98,
                            },
                            'flags': {},
                            'order': 2,
                            'mode': 0,
                            'outputs': [
                                {
                                    'name': 'MODEL',
                                    'type': 'MODEL',
                                    'links': [
                                        193,
                                    ],
                                    'shape': 3,
                                    'slot_index': 0,
                                },
                                {
                                    'name': 'CLIP',
                                    'type': 'CLIP',
                                    'links': [
                                        194,
                                    ],
                                    'shape': 3,
                                    'slot_index': 1,
                                },
                                {
                                    'name': 'VAE',
                                    'type': 'VAE',
                                    'links': [
                                        106,
                                    ],
                                    'shape': 3,
                                    'slot_index': 2,
                                },
                            ],
                            'properties': {
                                'Node name for S&R': 'CheckpointLoaderSimple',
                            },
                            'widgets_values': [
                                model
                            ],
                            'color': '#232',
                            'bgcolor': '#353',
                        },
                        {
                            'id': 50,
                            'type': 'LoRA Stacker',
                            'pos': [
                                -1189,
                                -108,
                            ],
                            'size': {
                                '0': 315,
                                '1': 226,
                            },
                            'flags': {},
                            'order': 3,
                            'mode': 0,
                            'inputs': [
                                {
                                    'name': 'lora_stack',
                                    'type': 'LORA_STACK',
                                    'link': None,
                                },
                            ],
                            'outputs': [
                                {
                                    'name': 'LORA_STACK',
                                    'type': 'LORA_STACK',
                                    'links': [
                                        145,
                                    ],
                                    'shape': 3,
                                    'slot_index': 0,
                                },
                            ],
                            'properties': {
                                'Node name for S&R': 'LoRA Stacker',
                            },
                            'widgets_values': [
                                'simple',
                                loralength] + lorainlist,
                            'color': '#222233',
                            'bgcolor': '#333355',
                            'shape': 1,
                        },
                        {
                            'id': 54,
                            'type': 'DARE Merge LoRA Stack',
                            'pos': [
                                -826,
                                -865,
                            ],
                            'size': {
                                '0': 315,
                                '1': 154,
                            },
                            'flags': {},
                            'order': 6,
                            'mode': 0,
                            'inputs': [
                                {
                                    'name': 'lora_stack',
                                    'type': 'LORA_STACK',
                                    'link': 145,
                                },
                            ],
                            'outputs': [
                                {
                                    'name': 'LoRA',
                                    'type': 'LoRA',
                                    'links': [
                                        142,
                                        146,
                                    ],
                                    'shape': 3,
                                    'slot_index': 0,
                                },
                            ],
                            'properties': {
                                'Node name for S&R': 'DARE Merge LoRA Stack',
                            },
                            'widgets_values': [
                                lambda_val,
                                p_val,
                                scale,
                                1,
                                'fixed',
                            ],
                            'color': '#223',
                            'bgcolor': '#335',
                        },
                        {
                            'id': 34,
                            'type': 'CR Seed',
                            'pos': [
                                -1677,
                                353,
                            ],
                            'size': {
                                '0': 315,
                                '1': 82,
                            },
                            'flags': {},
                            'order': 4,
                            'mode': 0,
                            'outputs': [
                                {
                                    'name': 'seed',
                                    'type': 'INT',
                                    'links': [
                                        68,
                                    ],
                                    'shape': 3,
                                    'slot_index': 0,
                                },
                            ],
                            'properties': {
                                'Node name for S&R': 'CR Seed',
                            },
                            'widgets_values': [
                                seed,
                                'fixed',
                            ],
                            'color': '#232',
                            'bgcolor': '#353',
                        },
                        {
                            'id': 48,
                            'type': 'Save LoRA',
                            'pos': [
                                1053,
                                -1172,
                            ],
                            'size': {
                                '0': 315,
                                '1': 58,
                            },
                            'flags': {},
                            'order': 7,
                            'mode': 0,
                            'inputs': [
                                {
                                    'name': 'LoRA',
                                    'type': 'LoRA',
                                    'link': 142,
                                },
                            ],
                            'title': 'Save LoRA (CTRM-M to unmute)',
                            'properties': {
                                'Node name for S&R': 'Save LoRA',
                            },
                            'widgets_values': [
                                'LoRAs/'+uuid,
                            ],
                            'color': '#322',
                            'bgcolor': '#533',
                        },
                        {
                            'id': 51,
                            'type': 'Float',
                            'pos': [
                                -463,
                                -59,
                            ],
                            'size': {
                                '0': 315,
                                '1': 58,
                            },
                            'flags': {},
                            'order': 5,
                            'mode': 0,
                            'outputs': [
                                {
                                    'name': 'FLOAT',
                                    'type': 'FLOAT',
                                    'links': [
                                        143,
                                        144,
                                    ],
                                    'shape': 3,
                                    'slot_index': 0,
                                },
                            ],
                            'title': 'Strength',
                            'properties': {
                                'Node name for S&R': 'LoRA strength',
                            },
                            'widgets_values': [
                                strength,
                            ],
                            'color': '#223',
                            'bgcolor': '#335',
                        },
                    ],
                    'links': [
                        [
                            20,
                            7,
                            0,
                            13,
                            2,
                            'CONDITIONING',
                        ],
                        [
                            23,
                            5,
                            0,
                            13,
                            5,
                            'LATENT',
                        ],
                        [
                            28,
                            13,
                            0,
                            8,
                            0,
                            'LATENT',
                        ],
                        [
                            54,
                            8,
                            0,
                            27,
                            0,
                            'IMAGE',
                        ],
                        [
                            62,
                            14,
                            0,
                            13,
                            3,
                            'SAMPLER',
                        ],
                        [
                            68,
                            34,
                            0,
                            13,
                            6,
                            'INT',
                        ],
                        [
                            82,
                            37,
                            0,
                            13,
                            4,
                            'SIGMAS',
                        ],
                        [
                            106,
                            41,
                            2,
                            8,
                            1,
                            'VAE',
                        ],
                        [
                            142,
                            54,
                            0,
                            48,
                            0,
                            'LoRA',
                        ],
                        [
                            143,
                            51,
                            0,
                            55,
                            3,
                            'FLOAT',
                        ],
                        [
                            144,
                            51,
                            0,
                            55,
                            4,
                            'FLOAT',
                        ],
                        [
                            145,
                            50,
                            0,
                            54,
                            0,
                            'LORA_STACK',
                        ],
                        [
                            146,
                            54,
                            0,
                            55,
                            0,
                            'LoRA',
                        ],
                        [
                            147,
                            56,
                            0,
                            57,
                            0,
                            'CONDITIONING',
                        ],
                        [
                            148,
                            49,
                            0,
                            57,
                            1,
                            'CONDITIONING',
                        ],
                        [
                            166,
                            57,
                            0,
                            13,
                            1,
                            'CONDITIONING',
                        ],
                        [
                            185,
                            55,
                            0,
                            37,
                            0,
                            'MODEL',
                        ],
                        [
                            186,
                            55,
                            0,
                            13,
                            0,
                            'MODEL',
                        ],
                        [
                            188,
                            55,
                            1,
                            56,
                            0,
                            'CLIP',
                        ],
                        [
                            189,
                            55,
                            1,
                            49,
                            0,
                            'CLIP',
                        ],
                        [
                            190,
                            55,
                            1,
                            7,
                            0,
                            'CLIP',
                        ],
                        [
                            193,
                            41,
                            0,
                            55,
                            1,
                            'MODEL',
                        ],
                        [
                            194,
                            41,
                            1,
                            55,
                            2,
                            'CLIP',
                        ],
                    ],
                    'groups': [
                        {
                            'title': 'Unmute (CTRL-M) if you want to save images.',
                            'bounding': [
                                1177,
                                -151,
                                536,
                                676,
                            ],
                            'color': '#3f789e',
                            'font_size': 24,
                            'locked': False,
                        },
                    ],
                    'config': {},
                    'extra': {},
                    'version': 0.4,
                    'seed_widgets': {
                        '34': 0,
                        '54': 3,
                    },
                },
            },
        },
    }


    response = requests.post('http://192.168.0.180:8189/prompt', headers=headers, json=json_data, verify=False)
    print(response.json())
    return response

def sample_image(i, imgdir,output_directory,loras, seed, prompt, run_id, savedir, model, lambda_value, p_value, steps, cfg, scale, strength):
    retries = 0
    sample_seed(loras, seed, prompt, run_id, savedir, model, lambda_value, p_value, steps, cfg, scale, strength)
    for filename in os.listdir(imgdir):
        print("Removing "+imgdir+"/"+filename)
        os.unlink(imgdir+"/"+filename)
    while True:
        retries += 1
        time.sleep(1)

        foundfiles = False
        for filename in os.listdir(imgdir):
            foundfiles = True
            shutil.move(os.path.join(imgdir, filename), output_directory+"/"+run_id+"/"+str(seed)+"_"+str(i)+".png")
        if foundfiles:
            print("Found")
            retries = 0
            break
        if retries > 1000:
            print("retry failure")
            retries = 0
            break


def main():
    parser = argparse.ArgumentParser(description='Generate images for evaluating a lora')
    parser.add_argument('-b', '--base', type=str, default="combo_eval", help='directory to read from')
    parser.add_argument('-p', '--prompt', type=str, default="__person__, __bg__", help='Prompt to use')
    parser.add_argument('-m', '--model', type=str, default="sdxl/mario/toprated1.safetensors", help='Base model to use')
    parser.add_argument('-np', '--negative_prompt', type=str, default="nsfw", help='negative prompt')
    parser.add_argument('-pval', '--p_value', type=float, default=0.3, help="between 0 and 1")
    parser.add_argument('-lval', '--lambda_value', type=float, default=1.5, help="1 - 2 likely")
    parser.add_argument('-cfg', '--cfg', type=float, default=1.0, help="sampler config")
    parser.add_argument('-s', '--steps', type=int, default=12, help="sampler steps")
    parser.add_argument('-x', '--scale', type=float, default=0.22, help="lora scale")
    parser.add_argument('-d', '--savedir', type=str, default="combo", help="directory to save images to")
    parser.add_argument('-str', '--strength', type=float, default=1.0, help="strength to apply the resulting lora with")
    parser.add_argument('--wildcards', help='path to load wildcards (usable with __wildcard__ like in a111)', type=str, required=False, default=None)
    args = parser.parse_args()

    # Randomly select files
    random_int = random.randint(3, 6)
    #random_int = random.randint(40, 50)
    loras = random.sample(filtered_files, random_int) if len(filtered_files) >= random_int else filtered_files

    # Print the selected files

    # Generate a UUID
    run_id = str(uuid.uuid4())
    output_directory = args.base

    # Prepare the data for JSON
    data_to_write = {
        'uuid': run_id,
        'loras': loras
    }

    # Write the data to a JSON file
    with open(f'{output_directory}/{run_id}.json', 'w') as outfile:
        json.dump(data_to_write, outfile, indent=4)

    print(f"Data written to {output_directory}/{run_id}.json")
    os.makedirs(output_directory+"/"+str(run_id), exist_ok=True)

    imgdir = "/ml2/trained/ComfyUI/output/"+args.savedir
    os.makedirs(imgdir, exist_ok=True)
    max_int_value = sys.maxsize
    for i in range(10):
        seed = random.randint(0, 1000000)
        prompt0 = ""
        for l in loras:
            prompt0 += l.split(".")[0]+", "
        prompt1 = wildcard_replace(args.prompt, args.wildcards)
        prompt2 = args.negative_prompt
        prompt = [prompt0,prompt1,prompt2]
        print("Sampling")
        print(prompt)
        print(args.steps, "steps")
        sample_image(i,imgdir,args.base,loras, seed, prompt, run_id, args.savedir, args.model, args.lambda_value, args.p_value, args.steps, args.cfg, args.scale, args.strength)
if __name__ == "__main__":
    main()

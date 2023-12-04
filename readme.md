# lora_eval

## Overview

`lora_eval` is a tool for evaluating LoRA (Low-Rank Adaptation) fine-tuned models using the automatic1111 Stable Diffusion API. It facilitates the comparison of different fine-tunes by generating and organizing output images.

## Requirements

- A trained lora compatible with the automatic1111 API
- A running instance of the automatic1111 API server

## Installation

1. Clone the `lora_eval` repository
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure the automatic1111 API server is running

## Usage

Run `evaluate_lora.py` with the necessary parameters to generate images:

```bash
python3 evaluate_lora.py -l "fancy_regular" -s 4.0 -t "fancy|regular"
```



Run `lora_anim.py` to generate video:
```bash
mkdir v5
mkdir anim2
CUDA_VISIBLE_DEVICES=0 python3 lora_anim.py -s -2.3 -e 2.3 -l "happy" -p " __sdprompt__, __bg__ BREAK happy, __person__" -np "blur, blurry" -n 2 -sd 1 -m 120 -url "http://192.168.0.180:7777/sdapi/v1/txt2img" -f "anim2" -b 200
```



---
language: 
- en
thumbnail: "images/Chiaroscuro_17_3.0.png"
widget:
- text: Chiaroscuro
  output:
    url: images/Chiaroscuro_17_3.0.png
- text: Chiaroscuro
  output:
    url: images/Chiaroscuro_19_3.0.png
- text: Chiaroscuro
  output:
    url: images/Chiaroscuro_20_3.0.png
- text: Chiaroscuro
  output:
    url: images/Chiaroscuro_21_3.0.png
- text: Chiaroscuro
  output:
    url: images/Chiaroscuro_22_3.0.png
tags:
- text-to-image
- stable-diffusion-xl
- lora
- template:sd-lora
- template:sdxl-lora
- sdxl-sliders
- ntcai.xyz-sliders
- concept
- diffusers
license: "mit"
inference: false
instance_prompt: "Chiaroscuro"
base_model: "stabilityai/stable-diffusion-xl-base-1.0"
---
# ntcai.xyz slider - Chiaroscuro (SDXL LoRA)

| Strength: -3 | Strength: 0 | Strength: 3 |
| --- | --- | --- |
| <img src="images/Chiaroscuro_17_-3.0.png" width=256 height=256 /> | <img src="images/Chiaroscuro_17_0.0.png" width=256 height=256 /> | <img src="images/Chiaroscuro_17_3.0.png" width=256 height=256 /> |
| <img src="images/Chiaroscuro_19_-3.0.png" width=256 height=256 /> | <img src="images/Chiaroscuro_19_0.0.png" width=256 height=256 /> | <img src="images/Chiaroscuro_19_3.0.png" width=256 height=256 /> |
| <img src="images/Chiaroscuro_20_-3.0.png" width=256 height=256 /> | <img src="images/Chiaroscuro_20_0.0.png" width=256 height=256 /> | <img src="images/Chiaroscuro_20_3.0.png" width=256 height=256 /> |


## Download

Weights for this model are available in Safetensors format.

## Trigger words

You can apply this LoRA with trigger words for additional effect:

```
Chiaroscuro
```

## Use in diffusers

```python
from diffusers import StableDiffusionXLPipeline
from diffusers import EulerAncestralDiscreteScheduler
import torch

pipe = StableDiffusionXLPipeline.from_single_file("https://huggingface.co/martyn/sdxl-turbo-mario-merge-top-rated/blob/main/topRatedTurboxlLCM_v10.safetensors")
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Load the LoRA
pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.Chiaroscuro', weight_name='Chiaroscuro.safetensors', adapter_name="Chiaroscuro")

# Activate the LoRA
pipe.set_adapters(["Chiaroscuro"], adapter_weights=[2.0])

prompt = "medieval rich kingpin sitting in a tavern, Chiaroscuro"
negative_prompt = "nsfw"
width = 512
height = 512
num_inference_steps = 10
guidance_scale = 2
image = pipe(prompt, negative_prompt=negative_prompt, width=width, height=height, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
image.save('result.png')
```

## Support the Patreon

If you like this model please consider [joining our Patreon](https://www.patreon.com/NTCAI).

By joining our Patreon, you'll gain access to an ever-growing library of over 1140+ unique and diverse LoRAs, covering a wide range of styles and genres. You'll also receive early access to new models and updates, exclusive behind-the-scenes content, and the powerful LoRA slider creator, allowing you to craft your own custom LoRAs and experiment with endless possibilities.

Your support on Patreon will allow us to continue developing and refining new models.

## Other resources

- [CivitAI](https://civitai.com/user/ntc) - Follow ntc on Civit for even more LoRAs
- [ntcai.xyz](https://ntcai.xyz) - See ntcai.xyz to find more articles and LoRAs

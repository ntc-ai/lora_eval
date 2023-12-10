from huggingface_hub import create_repo, Repository, delete_repo
import glob
import huggingface_hub
import os
import re
import shutil
import subprocess
import time

HF_NAME="ntc-ai/"

def count_files(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

def readme(slider_file, repo_name, slider_name, png_paths, repo_file, adapter_name, trigger_words, sliders_count, slider_details):
    base_model_repo = "https://huggingface.co/martyn/sdxl-turbo-mario-merge-top-rated"
    patreon_link = "https://www.patreon.com/NTCAI"
    widget_list = "\n".join([f'- text: {trigger_words}\n  output:\n    url: images/'+os.path.basename(png_path) for i, png_path in enumerate(png_paths)])
    pre_post_images = "| Strength: -3 | Strength: 0 | Strength: 3 |\n| --- | --- | --- |\n"
    for png_path in png_paths[:3]:
        img_path = os.path.splitext(os.path.basename(png_path))[0].split("_3.")[0]
        pre_post_images += "| "
        pre_post_images += f"<img src=\"images/{img_path}_-3.0.png\" width=256 height=256 />"
        pre_post_images += " | "
        pre_post_images += f"<img src=\"images/{img_path}_0.0.png\" width=256 height=256 />"
        pre_post_images += " | "
        pre_post_images += f"<img src=\"images/{img_path}_3.0.png\" width=256 height=256 />"
        pre_post_images += " |\n"
    return f"""
---
language: 
- en
thumbnail: "images/{png_paths[0]}"
widget:
{widget_list}
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
instance_prompt: "{trigger_words}"
base_model: "stabilityai/stable-diffusion-xl-base-1.0"
---
# ntcai.xyz slider - {trigger_words} (SDXL LoRA)

{pre_post_images}

## Download

Weights for this model are available in Safetensors format.

## Trigger words

You can apply this LoRA with trigger words for additional effect:

```
{trigger_words}
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
pipe.load_lora_weights('{HF_NAME}{repo_name}', weight_name='{repo_file}', adapter_name="{adapter_name}")

# Activate the LoRA
pipe.set_adapters(["{adapter_name}"], adapter_weights=[2.0])

prompt = "medieval rich kingpin sitting in a tavern, {trigger_words}"
negative_prompt = "nsfw"
width = 512
height = 512
num_inference_steps = 10
guidance_scale = 2
image = pipe(prompt, negative_prompt=negative_prompt, width=width, height=height, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
image.save('result.png')
```

## Support the Patreon

If you like this model please consider [joining our Patreon]({patreon_link}).

By joining our Patreon, you'll gain access to an ever-growing library of over {sliders_count}+ unique and diverse LoRAs, covering a wide range of styles and genres. You'll also receive early access to new models and updates, exclusive behind-the-scenes content, and the powerful LoRA slider creator, allowing you to craft your own custom LoRAs and experiment with endless possibilities.

Your support on Patreon will allow us to continue developing and refining new models.

## Other resources

- [CivitAI](https://civitai.com/user/ntc) - Follow ntc on Civit for even more LoRAs
- [ntcai.xyz](https://ntcai.xyz) - See ntcai.xyz to find more articles and LoRAs
"""

def upload_to_huggingface(slider_name, repo_name, local_repo_path, safetensors_file, safetensors_path, all_images, png_paths, adapter_name, trigger_words, slider_details, private=False):
    # Create a new repository on Hugging Face
    try:
        #delete_repo(HF_NAME+repo_name)
        repo = create_repo(HF_NAME+repo_name, private=private)
    except huggingface_hub.utils._errors.HfHubHTTPError as e:
        print("repo exists?", e)

    # Clone the repository locally
    repo = Repository(local_repo_path, clone_from=HF_NAME+repo_name, token=True)
    subprocess.run(["git", "remote", "set-url", "origin", f"git@hf.co:{HF_NAME}{repo_name}.git"], cwd=local_repo_path)
    subprocess.run(["git", "lfs", "install"], cwd=local_repo_path)
    subprocess.run(["git", "lfs", "track", "*.png"], cwd=local_repo_path)


    # Copy the safetensors file and PNGs into the cloned repository
    shutil.copy(safetensors_path, local_repo_path+'/'+adapter_name+".safetensors")

    os.makedirs(local_repo_path+"/images", exist_ok=True)
    for png_path in all_images:
        shutil.copy(png_path, local_repo_path+"/images")

    sliders_count = int((count_files("loras")-1)/10)*10

    # Generate a README file and link to the PNGs
    with open(os.path.join(local_repo_path, 'README.md'), 'w') as readme_file:
        readme_data = readme(slider_file=safetensors_path,
                repo_name=repo_name,
                slider_name=slider_name,
                png_paths=png_paths,
                repo_file=adapter_name+".safetensors",
                adapter_name=adapter_name,
                trigger_words=trigger_words,
                slider_details=slider_details,
                sliders_count=sliders_count)
        readme_file.write(readme_data)

    # Commit and push the changes
    repo.git_add(auto_lfs_track=True)
    repo.git_commit('Update README, safetensors and PNGs')
    print("Pushing to", repo)
    repo.git_push()


def create_repo_name(base_filename):
    # Filter out invalid characters
    repo_name = re.sub(' ', '-', base_filename)
    repo_name = re.sub(r'[^a-zA-Z0-9-_.]', '', repo_name)

    # Collapse sequences of dots and dashes
    repo_name = re.sub(r'\.{2,}', '.', repo_name)
    repo_name = re.sub(r'-{2,}', '-', repo_name)

    # Ensure it doesn't start or end with '-' or '.'
    repo_name = repo_name.strip('-.')

    # Ensure it doesn't end with '.git'
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]

    # Enforce max length
    if "." in repo_name:
        repo_name = "SDXL-LoRA-slider."+"".join(repo_name.split(".")[:-1])
    else:
        repo_name = "SDXL-LoRA-slider."+repo_name
    max_length = 96
    if len(repo_name) > max_length:
        repo_name = repo_name[:max_length]

    return repo_name

loras_path = 'evaluate/loras/'
evaluate_path = 'evaluate/'

if __name__ == "__main__":
    loras = os.listdir(loras_path)

    # Sort files by last modified time (newest last)
    sorted_loras = sorted(loras, key=lambda lora: os.path.getmtime(os.path.join(loras_path, lora)))

    for lora in sorted_loras:
        if lora.endswith('.safetensors'):
            safetensors_path = os.path.join(loras_path, lora)
            base_filename = os.path.splitext(lora)[0]

            # Pattern to match the specific png files
            img_pattern = os.path.join(evaluate_path, f'{base_filename}/*.png')
            all_images = sorted(glob.glob(img_pattern))
            png_paths = [
                path for path in all_images if "_3.0.png" in os.path.basename(path)
            ]

            repo_name = create_repo_name(base_filename)
            slider_details = {
                "unconditional": base_filename.split("...")[-1]
            }
            local_repo_path = f"repos/{repo_name}"
            if os.path.exists(local_repo_path):
                print(f"Repository exists, skipping: {local_repo_path}")
                continue
            trigger_words = base_filename.split("...")[0]
            adapter_name = trigger_words
            upload_to_huggingface(
                slider_name=repo_name,
                repo_name=repo_name,
                local_repo_path=local_repo_path,
                safetensors_path=os.path.join(loras_path, lora),
                safetensors_file=lora,
                slider_details = slider_details,
                adapter_name=adapter_name,
                trigger_words=trigger_words,
                all_images=all_images,
                png_paths=png_paths,
                private=False
            )
            print(f"Success! Repo is uploaded to {HF_NAME}{repo_name}")
            time.sleep(3600)

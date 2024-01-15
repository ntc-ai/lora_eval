import requests
import os
import json
import argparse
from lora_eval import lora_eval
import time


def download_and_process(lora_list_url, a111_url, lora_strengths):
    current_page=1

    # Directory for storing downloaded files
    download_dir = 'loras'
    os.makedirs(download_dir, exist_ok=True)

    while True:
        print(f"Getting list for page {current_page}")
        paginated_url = f"{lora_list_url}?page={current_page}&per_page=10"
        response = requests.get(paginated_url)
        # Fetching JSON data from the URL
        print("Parsing list")
        if response.status_code != 200:
            print("Failed to fetch data from the URL. Retry")
            print(response.status_code)
            time.sleep(5)
            continue
        current_page += 1

        data = response.json()
        # Check if there are no sliders to process
        if len(data['sliders']) == 0:
            print(f"No more data to process. Ending at page {current_page}.")
            break

        # Process each slider
        for slider in data['sliders']:
            download_url = slider['download_url']
            name = slider['name']
            prompts = slider['data']['prompts'][0]  # Assuming one prompt per slider
            positive = prompts['positive']
            target = prompts['target']
            neutral = prompts['neutral']
            unconditional = prompts['unconditional']
            if 'negative' in prompts:
                unconditional = prompts['negative']

            file_name = f"{positive}.{target}.{neutral}.{unconditional}.safetensors"
            file_name = file_name.replace("/",'-')
            file_path = os.path.join(download_dir, file_name)

            # Check if file already exists
            if os.path.exists(file_path):
                #print(f"File {file_path} already exists. Skipping download.")
                pass
            else:
                print("FNAME", file_name)
                # Download the file
                response = requests.get(download_url, timeout=180)
                if response.status_code == 200:
                    with open(file_path, 'wb') as file:
                        file.write(response.content)
                    print(f"Downloaded and saved {file_path}")
                else:
                    print(f"Failed to download {file_name}")
                    continue

            # Call the mock function
            unique_tags = list(set([positive, target, neutral, unconditional]))
            print("___________________", positive)
            lora_eval(a111_url, file_name.replace(".safetensors", ""), [positive.replace("/","-")], lora_strengths)
    print("Done parsing list")

def main():
    parser = argparse.ArgumentParser(description='Generate images for evaluating a lora')
    parser.add_argument('-s', '--lora_strength', nargs='*', required=True, help='lora strength, can be multiple: -2 0 2')
    parser.add_argument('-url', '--text_to_image_url', type=str, default="http://localhost:3000/sdapi/v1/txt2img", help='Url for text to image')
    parser.add_argument('-lora_url', '--lora_list_url', type=str, default="notpublic", help='Url to get lora list')
    args = parser.parse_args()

    download_and_process(args.lora_list_url, args.text_to_image_url, args.lora_strength)


if __name__ == '__main__':
    main()

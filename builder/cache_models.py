# builder/model_fetcher.py

import argparse
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL
import requests

def fetch_pretrained_model(model_class, model_name, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise

def download_model_from_civitai(civitai_key, model_id):
    '''
    Downloads a model from Civitai using the provided environment variables.
    '''
    if not civitai_key or not model_id:
        raise ValueError("CIVITAI_KEY and CIVITAI_MODEL_ID must be provided.")

    url = f"https://civitai.com/api/download/models/{model_id}?token={civitai_key}"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(f"models/model.safetensors", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Model {model_name} downloaded successfully.")
    else:
        raise Exception(f"Failed to download model: {response.status_code} - {response.text}")


def download_vae():
    common_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True
    }
    
    vae = fetch_pretrained_model(
        AutoencoderKL, "madebyollin/sdxl-vae-fp16-fix", **{"torch_dtype": torch.float16}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download models from Civitai.')
    parser.add_argument('--civitai_key', required=True, help='Civitai API key')
    parser.add_argument('--model_id', required=True, help='Civitai model ID')
    args = parser.parse_args()

    download_model_from_civitai(args.civitai_key, args.model_id)
    download_vae()

# builder/model_fetcher.py

import os
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

def download_model_from_civitai():
    '''
    Downloads a model from Civitai using the provided environment variables.
    '''
    civitai_key = os.getenv("CIVITAI_KEY")
    model_id = os.getenv("CIVITAI_MODEL_ID")
    model_name = os.getenv("CIVITAI_MODEL_NAME")

    if not civitai_key or not model_id or not model_name:
        raise ValueError("CIVITAI_KEY, CIVITAI_MODEL_ID, and CIVITAI_MODEL_NAME must be set in the environment.")

    url = f"https://civitai.com/api/download/models/{model_id}?token={civitai_key}"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(f"{model_name}.safetensors", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Model {model_name} downloaded successfully.")
    else:
        raise Exception(f"Failed to download model: {response.status_code} - {response.text}")
    '''
    Fetches the Stable Diffusion XL pipelines from the HuggingFace model hub.
    '''
    common_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True
    }
    
    pipe = fetch_pretrained_model(StableDiffusionXLPipeline,
                                  "stabilityai/stable-diffusion-xl-base-1.0", **common_args)
    vae = fetch_pretrained_model(
        AutoencoderKL, "madebyollin/sdxl-vae-fp16-fix", **{"torch_dtype": torch.float16}
    )
    print("Loaded VAE")
    refiner = fetch_pretrained_model(StableDiffusionXLImg2ImgPipeline,
                                     "stabilityai/stable-diffusion-xl-refiner-1.0", **common_args)

    return pipe, refiner, vae


def get_diffusion_pipelines():
    '''
    Fetches the Stable Diffusion XL pipelines from the HuggingFace model hub.
    '''
    common_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True
    }
    
    pipe = fetch_pretrained_model(StableDiffusionXLPipeline,
                                  "stabilityai/stable-diffusion-xl-base-1.0", **common_args)
    vae = fetch_pretrained_model(
        AutoencoderKL, "madebyollin/sdxl-vae-fp16-fix", **{"torch_dtype": torch.float16}
    )
    print("Loaded VAE")
    refiner = fetch_pretrained_model(StableDiffusionXLImg2ImgPipeline,
                                     "stabilityai/stable-diffusion-xl-refiner-1.0", **common_args)

    return pipe, refiner, vae


if __name__ == "__main__":
    download_model_from_civitai()
    get_diffusion_pipelines()

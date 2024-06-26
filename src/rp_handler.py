'''
Contains the handler function that will be called by the serverless.
'''

import a1111
import base64
import os
import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate
from rp_schemas import INPUT_SCHEMA


def generate_image(job):
    '''
    Generate an image from text using your Model
    '''
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")

    fname = a1111.gen(
        prompt=job_input['prompt'],
        neg=job_input['negative_prompt'],
        height=job_input['height'],
        width=job_input['width'],
        steps=job_input['num_inference_steps'],
        cfg_scale=job_input['guidance_scale']
        )
    print(fname)
    image_urls = []
    with open(fname, "rb") as image_file:
        image_data = base64.b64encode(
            image_file.read()).decode("utf-8")
        image_urls.append(f"data:image/png;base64,{image_data}")

    results = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input['seed']
    }

    return results


runpod.serverless.start({"handler": generate_image})

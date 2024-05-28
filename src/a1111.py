from auto1111sdk import StableDiffusionXLPipeline, civit_download
import sys
import uuid
import os

def gen(prompt, neg='', width=768, height=1024, steps=10, cfg_scale=7.5):
    output = pipe.generate_txt2img(prompt = prompt, negative_prompt=neg, cfg_scale=cfg_scale, height = height, width = width, steps = steps)

    fname = f"{uuid.uuid4()}.png"

    output[0].save(fname)
    return fname

modelurl = sys.argv[1]
modelname = sys.argv[2]
prompt = sys.argv[3]
neg = sys.argv[4]

if not os.path.exists(f"models/{modelname}.safetensors"):
    print(f"Model not found, downloading {modelurl} to {modelname}")
    civit_download(modelurl, f"models/{modelname}.safetensors")

pipe = StableDiffusionXLPipeline(f"models/{modelname}.safetensors")

for i in range(1, 5):
    fname = gen(prompt, neg=neg, cfg_scale=9, steps=25)
    print(fname)


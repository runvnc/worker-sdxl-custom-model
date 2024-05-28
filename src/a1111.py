from auto1111sdk import StableDiffusionXLPipeline
import sys
import uuid

pipe = StableDiffusionXLPipeline("models/model.safetensors")

def gen(prompt, neg='', width=768, height=1024, steps=10, cfg_scale=7.5):
    output = pipe.generate_txt2img(prompt = prompt, negative_prompt=neg, cfg_scale=cfg_scale, height = height, width = width, steps = steps)

    fname = f"{uuid.uuid4()}.png"

    output[0].save(fname)
    return fname

prompt = sys.argv[1]
neg = sys.argv[2]

for i in range(1, 5):
    fname = gen(prompt, neg=neg, cfg_scale=11, steps=27)
    print(fname)


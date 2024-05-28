from auto1111sdk import StableDiffusionXLPipeline
import sys
import uuid

pipe = StableDiffusionXLPipeline("models/model.safetensors")

def gen(prompt, neg, width, height, steps):
    output = pipe.generate_txt2img(prompt = prompt, negative_prompt=neg, height = 1024, width = 768, steps = 10)

    fname = f"{uuid.uuid4()}.png"

    output[0].save(fname)
    return fname

prompt = sys.argv[1]
neg = sys.argv[2]

for i in range 1..4:
    fname = gen(prompt, neg)
    print(fname)


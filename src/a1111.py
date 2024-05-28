from auto1111sdk import StableDiffusionXLPipeline
import sys

pipe = StableDiffusionXLPipeline("models/model.safetensors")

prompt = sys.argv[1]
neg = sys.argv[2]
output = pipe.generate_txt2img(prompt = prompt, negative_prompt=neg, height = 1024, width = 768, steps = 10)

output[0].save("image.png")

import json, os, tqdm, torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from JDiffusion.pipelines import StableDiffusionPipeline

save_root = "/data/jittor2024/Problem2/JDiffusion/examples/dreambooth/results/" + "prompt_v1_test1"

max_num = 15
dataset_root = "/data/jittor2024/Problem2/JDiffusion/A"
cached_path = "/data/jittor2024/Problem2/JDiffusion/cached_path"
style_file = "/data/jittor2024/Problem2/JDiffusion/examples/dreambooth/settings/style.json"
texture_file = "/data/jittor2024/Problem2/JDiffusion/examples/dreambooth/settings/texture.json"
color_file = "/data/jittor2024/Problem2/JDiffusion/examples/dreambooth/settings/color.json"


with open(style_file, "r") as f:
    style_dict = json.load(f)

with open(texture_file, "r") as f:
    texture_dict = json.load(f)

with open(color_file, "r") as f:
    color_dict = json.load(f)

with torch.no_grad():
    for tempid in tqdm.tqdm(range(0, max_num)):
        taskid = "{:0>2d}".format(tempid)
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")
        pipe.load_lora_weights(os.path.join(save_root, f"style/style_{taskid}"))

        # load json
        with open(f"{dataset_root}/{taskid}/prompt.json", "r") as file:
            prompts = json.load(file)

        for id, prompt in prompts.items():
            new_prompt = f"A photo of {prompt} in {style_dict[taskid]} style, with a texture of {texture_dict[taskid]}."
            # new_prompt = f"A photo of {prompt} in {style_dict[taskid]} style, with a texture of {texture_dict[taskid]} and with a color style of {color_dict[taskid]}."
            print(new_prompt)
            # image = pipe(prompt + f" in style_{taskid}", num_inference_steps=25, width=512, height=512).images[0]
            image = pipe(new_prompt, num_inference_steps=100, width=512, height=512).images[0]
            os.makedirs(os.path.join(save_root, f"outputs_100steps/{taskid}"), exist_ok=True)
            image.save(os.path.join(save_root, f"outputs_100steps/{taskid}/{prompt}.png"))



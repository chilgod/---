import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL
import os
import random


output_folder = "generated_doctor_2test"
os.makedirs(output_folder, exist_ok=True)


base_model_path = "./sdxl"
checkpoint_path = "./sd-naruto-model—meta-withvnet3/checkpoint-500/unet_ema" 
vae_path = f"{base_model_path}/vae"


unet = UNet2DConditionModel.from_pretrained(
    checkpoint_path,
    torch_dtype=torch.float16
)

vae = AutoencoderKL.from_pretrained(
    vae_path,
    torch_dtype=torch.float16
)


pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    unet=unet,
    vae=vae,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")


prompt = "a headshot of a doctor"
num_images = 100

for i in range(num_images):

    seed = random.randint(0, 1000000)
    generator = torch.Generator("cuda").manual_seed(seed)
    
    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            width=512,
            height=512,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
    

    image_path = os.path.join(output_folder, f"doctor_{i:03d}.png")
    image.save(image_path)
    print(f"已保存第 {i+1}/{num_images} 张图像: {image_path}")


    
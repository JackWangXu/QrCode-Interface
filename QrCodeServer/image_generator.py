from typing import Optional
import torch
from PIL import Image
import qrcode
from pathlib import Path
from multiprocessing import cpu_count
import requests
import io
import os
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    HeunDiscreteScheduler,
    EulerDiscreteScheduler,
)

# 初始化QRCode生成器
qrcode_generator = qrcode.QRCode(
    version=1,
    error_correction=qrcode.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)

current_script_path = Path(__file__).parent
qrcode_model_directory = current_script_path / 'control_v1p_sd15_qrcode'
sd_model_directory = current_script_path / 'stable-diffusion-v1-5'


# 加载ControlNet模型
controlnet = ControlNetModel.from_pretrained(str(qrcode_model_directory), torch_dtype=torch.float16, use_auth_token=False
)

# 加载StableDiffusionControlNetImg2ImgPipeline
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    str(sd_model_directory),
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,use_auth_token=False
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()


def resize_for_condition_image(input_image: Image.Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True,
                                                                               algorithm_type="sde-dpmsolver++"),
    "DPM++ Karras": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True),
    "Heun": lambda config: HeunDiscreteScheduler.from_config(config),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
    "DDIM": lambda config: DDIMScheduler.from_config(config),
    "DEIS": lambda config: DEISMultistepScheduler.from_config(config),
}


def inference(
        qr_code_content: str,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 2.0,
        strength: float = 0.5,
        seed: int = -1,
        init_image: Optional[Image.Image] = None,
        qrcode_image: Optional[Image.Image] = None,
        sampler="DPM++ Karras SDE",
):
    if qr_code_content == "" and qrcode_image is None:
        raise ValueError("QR Code Image or QR Code Content is required")

    pipe.scheduler = SAMPLER_MAP[sampler](pipe.scheduler.config)
    generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()

    if qrcode_image is None:
        print("Generating QR Code from content")
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_code_content)
        qr.make(fit=True)
        qrcode_image = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        qrcode_image = resize_for_condition_image(qrcode_image, 768)
        print(111111111111111111111)
        print(qrcode_image)
    else:
        print("Using uploaded QR Code Image")
        qrcode_image = resize_for_condition_image(qrcode_image, 768)
    print(6666666666666)
    if init_image is None:
        print("------init_image------------")
        init_image = qrcode_image

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        control_image=qrcode_image,
        width=768,
        height=768,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=generator,
        strength=strength,
        num_inference_steps=50,
    )

    print(out)
    return out["images"][0]

import numpy as np
from PIL import Image, ImageFilter, ImageChops, ImageOps
import torch
from torch import Tensor

import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.clip_vision
import comfy.model_management

import latent_preview

class KSampler:
    def __init__(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent
                 , denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
        self.model = model
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.schedular = scheduler
        self.positive = positive
        self.negative = negative
        self.latent = latent
        self.denoise = denoise
        self.disable_noise = disable_noise
        self.start_step = start_step
        self.last_step = last_step
        self.force_full_denoise = force_full_denoise

    def update_latent(self, latent):
        self.latent = latent

    def sample(self):
        latent_image = self.latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(self.model, latent_image)

        if self.disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = self.latent["batch_index"] if "batch_index" in self.latent else None
            noise = comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)

        noise_mask = None
        if "noise_mask" in self.latent:
            noise_mask = self.latent["noise_mask"]

        callback = latent_preview.prepare_callback(self.model, self.steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample(self.model, noise, self.steps, self.cfg, self.sampler_name, self.scheduler, self.positive, self.negative, latent_image,
                                    denoise=self.denoise, disable_noise=self.disable_noise, start_step=self.start_step, last_step=self.last_step,
                                    force_full_denoise=self.force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=self.seed)
        out = self.latent.copy()
        out["samples"] = samples
        return (out, )

# Node: Sample Image
class SampleImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "radius": ("INT", {"default": 10, "min": 1, "max": 500, "step": 1}),
                "strength": ("FLOAT", {"default": 4, "min": 0.0, "max": 255.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample_image"
    CATEGORY = "Albedo Sampler"

    def sample_image(self, images, radius=10, strength=4):
        for img in images:
            image = img
        hp_dark = 1.0 - images
        hp_dark = self.high_pass(hp_dark, radius, strength)
        hp_dark = 1.0 - hp_dark
        hp_light = self.high_pass(image, radius, strength)
        blend_tensor = self.image_blend(hp_dark, hp_light)
        ave_color = self.image_average_color(image)
        blend_tensor = self.image_blend_overlay(blend_tensor, ave_color)

        return (blend_tensor, )

    def high_pass(self, img, radius=10, strength=4):
        transformed_image = self.apply_hpf(tensor2pil(img), radius, strength)
        tensor = pil2tensor(transformed_image)

        return tensor

    def apply_hpf(self, img, radius=10, strength=4):
        img_arr = np.array(img).astype('float')
        blurred_arr = np.array(img.filter(ImageFilter.GaussianBlur(radius=radius))).astype('float')
        hpf_arr = img_arr - blurred_arr
        hpf_arr = np.clip(hpf_arr * strength, 0, 255).astype('uint8')
        high_pass = Image.fromarray(hpf_arr, mode='RGB')

        neutral_color = (128, 128, 128) if high_pass.mode == 'RGB' else 128
        neutral_bg = Image.new(high_pass.mode, high_pass.size, neutral_color)
        high_pass = ImageChops.screen(neutral_bg, high_pass)

        return high_pass.convert('RGB')
    
    def image_blend(self, image_a, image_b, blend_percentage=0.5):
        img_a = tensor2pil(image_a)
        img_b = tensor2pil(image_b)
        blend_mask = Image.new(mode="L", size=img_a.size, color=(round(blend_percentage * 255)))
        blend_mask = ImageOps.invert(blend_mask)
        img_result = Image.composite(img_a, img_b, blend_mask)

        return pil2tensor(img_result)
    
    def image_blend_overlay(self, image_a, image_b):
        img_a = tensor2pil(image_a)
        img_b = tensor2pil(image_b)    
        img_result = ImageChops.overlay(img_a, img_b)

        return pil2tensor(img_result)

    def image_average_color(self, img):
        pilimage = tensor2pil(img)
        pilimage.convert('RGB')
        width, height = pilimage.size
        total_red = 0
        total_blue = 0
        total_green = 0
        total_pixel = 0
        for x in range(width):
            for y in range(height):
                rgb = pilimage.getpixel((x,y))
                total_red += rgb[0]
                total_green += rgb[1]
                total_blue += rgb[2]
                total_pixel += 1
        ave_red = total_red // total_pixel
        ave_green = total_green // total_pixel
        ave_blue = total_blue // total_pixel
        ave_color_img = Image.new('RGB', (width,height), (ave_red,ave_green,ave_blue))

        return pil2tensor(ave_color_img)

# functions to convert betweem PIL Image and tensor
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Sample Image": SampleImage,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Sample Image": "Sample Image",
}

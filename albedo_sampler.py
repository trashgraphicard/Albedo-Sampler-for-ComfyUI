import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageChops, ImageOps
import torch

import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.clip_vision
import comfy.model_management

import latent_preview

class KSAMPLER:
    def __init__(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent
                 , denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
        self.model = model
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
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
        return out

# Node: Sample Image
class SampleImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
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
# Node: Make Seamless Tile
class MakeSeamlessTile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "mask_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "kernel_size": ("INT", {"default": 79, "min": 1, "max": 10000}),
                "sigma": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = ("make_seamless_tile")
    CATEGORY = ("Albedo Sampler")

    def make_seamless_tile(self, model, vae, positive, negative, mask_strength, kernel_size, sigma, image, seed, steps, cfg, sampler_name, scheduler, denoise):
        
        cross_mask = self.make_cross_mask(image, mask_strength, kernel_size, sigma)
        oval_mask = self.make_oval_mask(image, mask_strength, kernel_size, sigma)

        # Offset image x and y and inpaint the seams
        base_img = self.offset_image(image)
        latent = self.vaeencode(base_img, vae)
        latent = self.set__latent_noise_mask(latent, cross_mask)
        ksampler = KSAMPLER(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise)
        latent = ksampler.sample()
        img = self.vaedecode(latent, vae)
        img = self.composite_tensor_image(img, base_img, cross_mask)

        # Offset image x and inpaint the seam results from first inpaint
        base_img = self.offset_image(img, 0, 0.5)
        latent = self.vaeencode(base_img, vae)
        latent = self.set__latent_noise_mask(latent, oval_mask)
        ksampler.update_latent(latent)
        latent = ksampler.sample()
        img = self.vaedecode(latent, vae)
        img = self.composite_tensor_image(img, base_img, oval_mask)

        # Offset image y and inpaint the seam results from first inpaint
        base_img = self.offset_image(img, 0.5, 0.5)
        latent = self.vaeencode(base_img, vae)
        latent = self.set__latent_noise_mask(latent, oval_mask)
        ksampler.update_latent(latent)
        latent = ksampler.sample()
        img = self.vaedecode(latent, vae)
        img = self.composite_tensor_image(img, base_img, oval_mask)

        # Offset image back to its original position
        img = self.offset_image(img, 0, 0.5)

        return (img,)

    def vaeencode(self, image, vae):
        lat = vae.encode(image[:,:,:,:3])
        return {"samples": lat}
    
    def vaedecode(self, latent, vae):
        images = vae.decode(latent["samples"])
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images
    
    def set__latent_noise_mask(self, samples, mask):
        s = samples.copy()
        s["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        return s
    
    def offset_image(self, image, offset_x = 0.5, offset_y = 0.5):

        # 4 dimension tensor, batch, height, width, channel
        B, H, W, C = image.shape

        offset_x = int((offset_x % 1.0) * H)  # Wrap-around for Y-axis offset
        offset_y = int((offset_y % 1.0) * W)  # Wrap-around for X-axis offset
        output_image = torch.zeros_like(image)

        # Slicing
        output_image[:, :H-offset_x, :W-offset_y, :] = image[:, offset_x:, offset_y:, :]
        output_image[:, :H-offset_x, W-offset_y:, :] = image[:, offset_x:, :offset_y, :]
        output_image[:, H-offset_x:, :W-offset_y, :] = image[:, :offset_x, offset_y:, :]
        output_image[:, H-offset_x:, W-offset_y:, :] = image[:, :offset_x, :offset_y, :]

        return output_image
    
    def make_cross_mask(self, image, mask_strength, kernel_size, sigma):
        B, H, W, C = image.shape
        if not (0 <= mask_strength <= 1):
            raise ValueError("Strength must be between 0 and 1.")
    
        vertical_thickness = int(H * mask_strength)
        horizontal_thickness = int(W * mask_strength)
        mask = np.zeros((H, W), dtype=np.uint8)
    
        center_y, center_x = H // 2, W // 2
        mask[center_y - vertical_thickness // 2:center_y + vertical_thickness // 2, :] = 255
        mask[:, center_x - horizontal_thickness // 2:center_x + horizontal_thickness // 2] = 255

        # blur the mask then normalize it (divide by 255)
        normalized_blurred_mask = self._gaussian_blur_mask(mask, kernel_size, sigma) / 255.0
    
        return torch.tensor(normalized_blurred_mask, dtype=torch.float32)

    def make_oval_mask(self, image, mask_strength, kernel_size, sigma):
        
        B, H, W, C = image.shape
    
        # Calculate the semi-major and semi-minor axes of the oval
        a = int(W * mask_strength / 2)
        b = int(H * mask_strength / 2)

        mask = np.zeros((H, W), dtype=np.uint8)

        # Generate the oval mask
        y, x = np.ogrid[:H, :W]
        center_y, center_x = H // 2, W // 2
        ellipse_mask = ((x - center_x) ** 2) / (a ** 2) + ((y - center_y) ** 2) / (b ** 2) <= 1
        mask[ellipse_mask] = 255

        # Blur the mask and normalize it (divide by 255)
        normalized_blurred_mask = self._gaussian_blur_mask(mask, kernel_size, sigma) / 255.0

        return torch.tensor(normalized_blurred_mask, dtype=torch.float32)
    
    def _gaussian_blur_mask(self, mask, kernel_size, sigma = 10.0):
        if kernel_size <= 0:
            raise ValueError("Strength must be a positive.")
        # Strength mush be odd number due to kernel
        if kernel_size % 2 == 0:
            kernel_size += 1
    
        blurred_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
    
        return blurred_mask
    
    def composite_tensor_image(self, img1, img2, mask):

        mask = mask.unsqueeze(0).unsqueeze(-1)
        mask = mask.expand(img1.shape[0], -1, -1, img1.shape[-1])

        return mask * img1 + (1 - mask) * img2


# functions to convert betweem PIL Image and tensor
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Sample Image": SampleImage,
    "Make Seamless Tile": MakeSeamlessTile
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Sample Image": "Sample Image",
    "Make Seamless Tile": "Make Seamless Tile"
}

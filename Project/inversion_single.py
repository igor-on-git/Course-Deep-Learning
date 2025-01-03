import torch
import numpy as np

from PIL import Image, ImageFilter

from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig

from main import run as invert

def inversion_single(source_file_path, config_run, model_type, scheduler_type, pipe_inversion, pipe_inference, num_inversion_steps=20, num_inference_steps=20, inversion_max_step=1.0):

    input_image = Image.open(source_file_path)
    resized_image = input_image.convert("RGB").resize(config_run.pic_dim)

    if config_run.filt_rad>0.0:
        filtered_image = resized_image.filter(ImageFilter.GaussianBlur(radius = config_run.filt_rad))
    else:
        filtered_image = resized_image

    if config_run.model_name == 'SD15':
        config = RunConfig(model_type = model_type,
                            num_inference_steps = num_inference_steps,
                            num_inversion_steps = num_inversion_steps,
                            inversion_max_step = inversion_max_step,
                            num_renoise_steps = 0,
                            scheduler_type = scheduler_type,
                            perform_noise_correction = False,
                            seed = config_run.seed)
    else:
        config = RunConfig(model_type=model_type,
                           num_inference_steps = num_inference_steps,
                           num_inversion_steps = num_inversion_steps,
                           inversion_max_step=inversion_max_step,
                           num_renoise_steps=0,
                           scheduler_type=scheduler_type,
                           perform_noise_correction=False,
                           seed = config_run.seed)

    _, inv_latent, _, all_latents = invert(filtered_image,
                                           config_run.prompt,
                                           config,
                                           pipe_inversion=pipe_inversion,
                                           pipe_inference=pipe_inference,
                                           do_reconstruction=False)
    if config_run.model_name == 'SD15':
        rec_image = pipe_inference(image = inv_latent,
                                   prompt = config_run.prompt,
                                   strength=inversion_max_step,
                                   num_inference_steps = config.num_inference_steps,
                                   guidance_scale = 1.0).images[0]
    else:
        rec_image = pipe_inference(image=inv_latent,
                                   prompt=config_run.prompt,
                                   denoising_start=1.0 - inversion_max_step,
                                   num_inference_steps=config.num_inference_steps,
                                   guidance_scale=1.0).images[0]

    return input_image, resized_image, filtered_image, rec_image

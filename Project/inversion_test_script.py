import torch
import numpy as np

from PIL import Image

from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig

from main import run as invert

#source_folder_path = 'C:/Users/FileServer/PycharmProjects/ReNoise-Inversion-main/example_images/'
#file_name = 'lion.jpeg'; prompt = "a lion in the field"

model_name = 'SDXL' #'SD15' #

#source_file_path = './../data/lfw_funneled/Abel_Pacheco/'; source_file_name = 'Abel_Pacheco_0004'
#source_file_path = './../data/lfw_funneled/Abel_Pacheco/'; source_file_name = 'Abel_Pacheco_0004_clear'
#source_file_path = './../data/lfw_funneled/Abel_Pacheco/'; source_file_name = 'Abel_Pacheco_0004_fgsm0.020_same_0_0'
#source_file_path = './../data/lfw_funneled/Abel_Pacheco/'; source_file_name = 'Abel_Pacheco_0004_fgsm0.020_same_0_0_full'
#source_file_path = './../data/lfw_funneled/Akhmed_Zakayev/'; source_file_name = 'Akhmed_Zakayev_0003'
#source_file_path = './../data/lfw_funneled/Akhmed_Zakayev/'; source_file_name = 'Akhmed_Zakayev_0003_clear'
#source_file_path = './../data/lfw_funneled/Akhmed_Zakayev/'; source_file_name = 'Akhmed_Zakayev_0003_fgsm0.020_same_0_1'
source_file_path = './../data/lfw_funneled/Akhmed_Zakayev/'; source_file_name = 'Akhmed_Zakayev_0003_fgsm0.020_same_0_1_full'

destination_file_path = './processed.jpg'
prompt = "a photo of a person"

shape = (512 ,512)
num_inversion_steps_vec = [5, 10, 20, 40]
inversion_strength_vec = [0.25, 0.5, 1.0]
num_renoise_steps = 0
noise_reg_lambda_ac = 20.0*1.5
noise_reg_lambda_kl = 0.065*1.5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if model_name == 'SD15':
    model_type = Model_Type.SD15
elif model_name == 'SDXL':
    model_type = Model_Type.SDXL

scheduler_type = Scheduler_Type.DDIM
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)

input_image = Image.open(source_file_path + source_file_name +'.jpg').convert("RGB").resize(shape)

combined_arr = np.empty((0, shape[1]*len(inversion_strength_vec), 3), dtype='uint8')
for num_inversion_steps in num_inversion_steps_vec:
    num_inference_steps = num_inversion_steps
    combined_inner_loop_arr = np.empty((shape[0], 0, 3), dtype='uint8')
    for inversion_strength in inversion_strength_vec:
        if model_name == 'SD15':
            config = RunConfig(model_type = model_type,
                                num_inference_steps = num_inference_steps,
                                num_inversion_steps = num_inversion_steps,
                                inversion_max_step = inversion_strength,
                                num_renoise_steps = num_renoise_steps,
                                scheduler_type = scheduler_type,
                                perform_noise_correction = False,
                                noise_regularization_lambda_ac = noise_reg_lambda_ac,
                                noise_regularization_lambda_kl = noise_reg_lambda_kl,
                                seed = 7865)
        elif model_name == 'SDXL':
            config = RunConfig(model_type = model_type,
                                num_inference_steps = num_inference_steps,
                                num_inversion_steps = num_inversion_steps,
                                inversion_max_step = inversion_strength,
                                num_renoise_steps = num_renoise_steps,
                                scheduler_type = scheduler_type,
                                perform_noise_correction = False,
                                noise_regularization_lambda_ac=noise_reg_lambda_ac,
                                noise_regularization_lambda_kl=noise_reg_lambda_kl,
                                seed = 7865)

        _, inv_latent, _, all_latents = invert(input_image,
                                               prompt,
                                               config,
                                               pipe_inversion=pipe_inversion,
                                               pipe_inference=pipe_inference,
                                               do_reconstruction=False)

        if model_name == 'SD15':
            rec_image = pipe_inference(image = inv_latent,
                                       prompt = prompt,
                                       strength=inversion_strength,
                                       num_inference_steps = config.num_inference_steps,
                                       guidance_scale = 1.0).images[0]
        elif model_name == 'SDXL':
            rec_image = pipe_inference(image = inv_latent,
                                       prompt = prompt,
                                       denoising_start=1.0-inversion_strength,
                                       num_inference_steps = config.num_inference_steps,
                                       guidance_scale = 1.0).images[0]

        combined_inner_loop_arr = np.concatenate((combined_inner_loop_arr,np.array(rec_image)), axis=1)

    combined_arr = np.concatenate((combined_arr, combined_inner_loop_arr), axis=0)

combined = Image.fromarray(combined_arr)
combined.show()
if num_renoise_steps>0:
    combined.save('./' + source_file_name + '_ac_' + str(noise_reg_lambda_ac) + '_kl_' + str(noise_reg_lambda_kl) + '_rn' + str(num_renoise_steps) + '.jpg')
else:
    combined.save('./' + source_file_name + '_ac_' + str(noise_reg_lambda_ac) + '_kl_' + str(noise_reg_lambda_kl) + '.jpg')
exit(1)

#input_image.save('./input_image.jpg')
#rec_image.save(destination_file_path)

# side by side
input_image_arr = np.array(input_image)
rec_image_arr = np.array(rec_image)

combined_arr = np.vstack((input_image_arr,rec_image_arr))


#combined.save('./compare_image.jpg')
combined.show()
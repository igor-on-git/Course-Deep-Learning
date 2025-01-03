import torch
import numpy as np

from PIL import Image

from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig

from main import run as invert
from inversion_single import inversion_single

#source_folder_path = 'C:/Users/FileServer/PycharmProjects/ReNoise-Inversion-main/example_images/'
#file_name = 'lion.jpeg'; prompt = "a lion in the field"

source_folder_path = 'F:/PycharmProjects/CelebA-HQ/'
destination_folder_path = 'F:/PycharmProjects/CelebA-HQ/processed/'
file_name = '55.jpg'; prompt = "a picture of a woman"
#file_name = '1946.jpg'; prompt = "a picture of a man"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_inference_steps = np.asarray([1, 2, 4, 8, 12, 16]) #np.asarray([1, 2, 4, 6, 8, 10, 12, 14, 16, 18])
num_inversion_steps = 20*np.ones(len(num_inference_steps), dtype=np.int64)

seed = 2

model_type = Model_Type.SD15
scheduler_type = Scheduler_Type.DDIM
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)

source_file_path = source_folder_path + file_name

for i in range(len(num_inference_steps)):
    input_image, rec_image = inversion_single(source_file_path, prompt, model_type, scheduler_type, pipe_inversion, pipe_inference, num_inversion_steps[i], num_inference_steps[i], seed)
    destination_file_path = destination_folder_path + file_name[:-4] + '_' + str(num_inversion_steps[i]) + '_' + str(num_inference_steps[i]) + file_name[-4:]
    rec_image.save(destination_file_path)

    rec_image_arr = np.array(rec_image)
    if i == 0:
        input_image_arr = np.array(input_image)
        combined_arr = np.vstack((input_image_arr,rec_image_arr))
    else:
        combined_arr = np.vstack((combined_arr, rec_image_arr))

combined = Image.fromarray(combined_arr)
combined.show()
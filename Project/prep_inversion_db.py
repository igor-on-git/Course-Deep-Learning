import torch
import numpy as np
import os

from PIL import Image

from torchvision.datasets import LFWPeople

from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes

from read_pairs_lfw import read_pairs_lfw
from inversion_single import inversion_single

prompt = "picture of a person"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = 'SDXL' #'SD15'
source_folder = './../data/lfw_funneled'
destination_folder = './../data/lfw_funneled_inv_clean'

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

num_inference_steps = 10
num_inversion_steps = num_inference_steps
strength = 0.5
seed = 2

if model_name == 'SD15':
    model_type = Model_Type.SD15
elif model_name == 'SDXL':
    model_type = Model_Type.SDXL

scheduler_type = Scheduler_Type.DDIM
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)

# lfw_ds = LFWPeople(root='./../data/', download=True)

(same_pairs_name_sets, diff_pairs_names_sets, same_pairs_numbers_sets, diff_pairs_numbers_sets, nsets, npairs) = read_pairs_lfw(source_folder)

rerun = [1, 0]
nsets = 1
#pairs_vec = np.arange(0,npairs,dtype=np.int64)
pairs_vec = np.arange(0,31,dtype=np.int64)
for set_cnt in range(nsets):
    if rerun[0]:
        # SAME PERSON
        same_dist = []
        same_dist_dp = []
        for npair in pairs_vec:
            if (npair + 1) % 20 == 0:
                print(npair + 1)
            image1_path = source_folder + '/{:s}/{:s}_{:04d}.jpg'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 0])
            image2_path = source_folder + '/{:s}/{:s}_{:04d}.jpg'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 1])
            image2_path_inv = destination_folder + '/numsteps_{:02d}_strength_{:0.2f}/{:s}/{:s}_{:04d}.jpg'.format(num_inversion_steps, strength, same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 1])

            path = destination_folder + '/numsteps_{:02d}_strength_{:0.2f}'.format(num_inversion_steps, strength)
            if not os.path.exists(path):
                os.mkdir(path)
            path = destination_folder + '/numsteps_{:02d}_strength_{:0.2f}/{:s}'.format(num_inversion_steps,strength,same_pairs_name_sets[set_cnt][npair])
            if not os.path.exists(path):
                os.mkdir(path)
            #img1 = Image.open(image1_path)
            #img2 = Image.open(image2_path)

            input_image, rec_image = inversion_single(image2_path, prompt, model_type, scheduler_type,
                                                      pipe_inversion, pipe_inference, num_inversion_steps=num_inversion_steps,
                                                      num_inference_steps=num_inference_steps, inversion_max_step=strength, seed=seed)
            rec_image.save(image2_path_inv)

    if rerun[1]:
        diff_dist = []
        # DIFF PERSON
        for npair in range(npairs):
            if (npair + 1) % 20 == 0:
                print(npair + 1)
            image1_path = source_folder + '/{:s}/{:s}_{:04d}.jpg'.format(diff_pairs_names_sets[set_cnt][npair][0], diff_pairs_names_sets[set_cnt][npair][0], diff_pairs_numbers_sets[set_cnt, npair, 0])
            image2_path = source_folder + '/{:s}/{:s}_{:04d}.jpg'.format(diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_numbers_sets[set_cnt, npair, 1])
            img1 = Image.open(image1_path)
            img2 = Image.open(image2_path)

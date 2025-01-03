import torch
import numpy as np
import os

from PIL import Image, ImageFilter

from torchvision.datasets import LFWPeople

from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes

from read_pairs_lfw import read_pairs_lfw
from inversion_single import inversion_single

import config_run

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if config_run.model_name == 'SD15':
    model_type = Model_Type.SD15
else:
    model_type = Model_Type.SDXL

scheduler_type = Scheduler_Type.DDIM
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)

# lfw_ds = LFWPeople(root='./../data/', download=True)

(same_pairs_name_sets, diff_pairs_names_sets, same_pairs_numbers_sets, diff_pairs_numbers_sets, nsets, npairs) = read_pairs_lfw(config_run.source_folder)

#nsets = 1
pairs_vec = np.arange(30,npairs,dtype=np.int64)
for fgsm_perturb_weight in config_run.fgsm_perturb_weight_vec:
    for num_inversion_steps in config_run.num_inversion_steps_vec:
        num_inference_steps = num_inversion_steps
        for inv_strength in config_run.inv_strength_vec:
            for set_cnt in config_run.sets_vec:
                if config_run.rerun[0]:
                    # SAME PERSON
                    same_dist = []
                    same_dist_dp = []
                    for npair in range(npairs):
                        if (npair + 1) % 20 == 0:
                            print(npair + 1)
                        image1_path = config_run.source_folder + '/{:s}/{:s}_{:04d}.jpg'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 0])
                        image2_path = config_run.source_folder + '/{:s}/{:s}_{:04d}.jpg'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 1])
                        if config_run.read_full_file:
                            image2_fgsm_same = config_run.source_folder + '/{:s}/{:s}_{:04d}_fgsm{:.3f}_same_{:d}_{:d}_full.jpg'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 1], fgsm_perturb_weight, set_cnt, npair)
                            image2_path_filt = config_run.destination_folder + '/{:s}/{:s}_{:04d}_fgsm{:.3f}_same_{:d}_{:d}_filt_full.jpg'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 1], fgsm_perturb_weight, set_cnt, npair)
                            image2_path_inv = config_run.destination_folder + '/{:s}/{:s}_{:04d}_fgsm{:.3f}_same_{:d}_{:d}_inv_{:s}_{:02d}_{:03d}_full.jpg'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 1], fgsm_perturb_weight, set_cnt, npair, config_run.model_name, num_inversion_steps, int(inv_strength*100))
                        else:
                            image2_fgsm_same = config_run.source_folder + '/{:s}/{:s}_{:04d}_fgsm{:.3f}_same_{:d}_{:d}.jpg'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 1], fgsm_perturb_weight, set_cnt, npair)
                            image2_path_filt = config_run.destination_folder + '/{:s}/{:s}_{:04d}_fgsm{:.3f}_same_{:d}_{:d}_filt.jpg'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 1], fgsm_perturb_weight, set_cnt, npair)
                            image2_path_inv = config_run.destination_folder + '/{:s}/{:s}_{:04d}_fgsm{:.3f}_same_{:d}_{:d}_inv_{:s}_{:02d}_{:03d}.jpg'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 1], fgsm_perturb_weight, set_cnt, npair, config_run.model_name, num_inversion_steps, int(inv_strength*100))
                        os.makedirs(config_run.destination_folder + '/{:s}'.format(same_pairs_name_sets[set_cnt][npair]), exist_ok=True)

                        if config_run.run_clean:
                            image_clean_path_filt = config_run.destination_folder + '/{:s}/{:s}_{:04d}_filt.jpg'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 1])
                            image_clean_path_inv = config_run.destination_folder + '/{:s}/{:s}_{:04d}_filt_inv_{:s}_{:02d}_{:03d}.jpg'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 1], config_run.model_name, num_inversion_steps, int(inv_strength*100))

                            if config_run.overwrite_files or not os.path.exists(image_clean_path_inv):
                                input_image, resized_image, filtered_image, rec_image = inversion_single(image2_path, config_run, model_type, scheduler_type,
                                                                      pipe_inversion, pipe_inference, num_inversion_steps=num_inversion_steps,
                                                                      num_inference_steps=num_inference_steps, inversion_max_step=inv_strength)
                                filtered_image.save(image_clean_path_filt)
                                rec_image.save(image_clean_path_inv)

                        if config_run.run_fgsm:
                            if config_run.overwrite_files or not os.path.exists(image2_path_inv):
                                input_image, resized_image, filtered_image, rec_image = inversion_single(image2_fgsm_same, config_run, model_type, scheduler_type,
                                                                          pipe_inversion, pipe_inference, num_inversion_steps=num_inversion_steps,
                                                                          num_inference_steps=num_inference_steps, inversion_max_step=inv_strength)
                                filtered_image.save(image2_path_filt)
                                rec_image.save(image2_path_inv)

                if config_run.rerun[1]:
                    diff_dist = []
                    # DIFF PERSON
                    for npair in range(npairs):
                        if (npair + 1) % 20 == 0:
                            print(npair + 1)
                        image1_path = config_run.source_folder + '/{:s}/{:s}_{:04d}.jpg'.format(diff_pairs_names_sets[set_cnt][npair][0], diff_pairs_names_sets[set_cnt][npair][0], diff_pairs_numbers_sets[set_cnt, npair, 0])
                        image2_path = config_run.source_folder + '/{:s}/{:s}_{:04d}.jpg'.format(diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_numbers_sets[set_cnt, npair, 1])
                        if config_run.read_full_file:
                            image2_fgsm_diff = config_run.source_folder + '/{:s}/{:s}_{:04d}_fgsm{:.3f}_diff_{:d}_{:d}_full.jpg'.format(diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_numbers_sets[set_cnt, npair, 1], fgsm_perturb_weight, set_cnt, npair)
                            image2_path_filt = config_run.destination_folder + '/{:s}/{:s}_{:04d}_fgsm{:.3f}_diff_{:d}_{:d}_filt_full.jpg'.format(diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_numbers_sets[set_cnt, npair, 1], fgsm_perturb_weight, set_cnt, npair)
                            image2_path_inv =  config_run.destination_folder + '/{:s}/{:s}_{:04d}_fgsm{:.3f}_diff_{:d}_{:d}_inv_{:s}_{:02d}_{:03d}_full.jpg'.format(diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_numbers_sets[set_cnt, npair, 1], fgsm_perturb_weight, set_cnt, npair, config_run.model_name, num_inversion_steps, int(inv_strength*100))
                        else:
                            image2_fgsm_diff = config_run.source_folder + '/{:s}/{:s}_{:04d}_fgsm{:.3f}_diff_{:d}_{:d}.jpg'.format(diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_numbers_sets[set_cnt, npair, 1], fgsm_perturb_weight, set_cnt, npair)
                            image2_path_filt = config_run.destination_folder + '/{:s}/{:s}_{:04d}_fgsm{:.3f}_diff_{:d}_{:d}_filt.jpg'.format(diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_numbers_sets[set_cnt, npair, 1], fgsm_perturb_weight, set_cnt, npair)
                            image2_path_inv =  config_run.destination_folder + '/{:s}/{:s}_{:04d}_fgsm{:.3f}_diff_{:d}_{:d}_inv_{:s}_{:02d}_{:03d}.jpg'.format(diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_numbers_sets[set_cnt, npair, 1], fgsm_perturb_weight, set_cnt, npair, config_run.model_name, num_inversion_steps, int(inv_strength*100))
                        os.makedirs(config_run.destination_folder + '/{:s}'.format(diff_pairs_names_sets[set_cnt][npair][1]), exist_ok=True)

                        if config_run.run_fgsm:
                            if config_run.overwrite_files or not os.path.exists(image2_path_inv):
                                input_image, resized_image, filtered_image, rec_image = inversion_single(image2_fgsm_diff, config_run, model_type, scheduler_type,
                                                                          pipe_inversion, pipe_inference, num_inversion_steps=num_inversion_steps,
                                                                          num_inference_steps=num_inference_steps, inversion_max_step=inv_strength)
                                filtered_image.save(image2_path_filt)
                                rec_image.save(image2_path_inv)

os.system('cp ./config_run.py ' + config_run.destination_folder)
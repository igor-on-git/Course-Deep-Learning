model_name = 'SDXL' #'SD15'
prompt = "picture of a person"
rerun = [1, 1] # run same and diff loops
sets_vec = [0]
run_clean = 1 # save on same set resized and filtered image
run_fgsm = 1 # run on fgsm photos
fgsm_perturb_weight_vec = [0.02]
num_inversion_steps_vec = [10] #[10, 30, 50]
inv_strength_vec = [0.5] #[0.5, 1.0]
read_full_file = 1
overwrite_files = 0
pic_dim = (512, 512)
filt_rad = 4.0
file_suffix = 'filt' + str(filt_rad)

source_folder = './../data/lfw_funneled'
destination_folder = source_folder + '_inv_{:s}_{:s}_pm1'.format(model_name, file_suffix)

seed = 7865


'''
prompt = "picture of a person"
model_name = 'SDXL' #'SD15'
source_folder = './../data/lfw_funneled'
destination_folder = './../data/lfw_funneled_inv_clean'
num_inference_steps = 10
num_inversion_steps = num_inference_steps
strength = 0.5
seed = 7865
rerun = [1, 0]
nsets = 1

'''

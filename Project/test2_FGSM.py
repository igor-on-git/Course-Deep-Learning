# 2024Aug16
# Face Recognition Diffusion Model (FRDM) Project
# This test gathers accuracy statistics of facenet on LFW dataset

# Based on:
# https://medium.com/@danushidk507/facenet-pytorch-pretrained-pytorch-face-detection-mtcnn-and-facial-recognition-b20af8771144
# https://pub.aimind.so/a-minimal-example-of-face-recognition-and-facial-analysis-ce4024da30d8
# pairs.txt downloaded from: https://vis-www.cs.umass.edu/lfw/#views
# pairs README: https://vis-www.cs.umass.edu/lfw/README.txt

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision.datasets import LFWPeople
import numpy as np
from matplotlib import pyplot as plt
import os
import torch
import torchvision
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchvision.io import read_image
import time

test_case = 1   # 0 or 1 (different or same)
mydir = './data/'
dataset_path = 'F:/PycharmProjects/lfw_funneled'
nsets = 10
npairs = 300


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def read_pairs_lfw():
    f = open('data/pairs.txt', "r")
    pairs_file = f.readlines()
    firstline = pairs_file[0].strip('\n').split('\t')
    assert (len(firstline) == 2)
    # nsets, npairs = int(firstline[0]), int(firstline[1])
    line_cnt = 0
    same_pairs_name_sets = []
    diff_pairs_names_sets = []
    same_pairs_numbers_sets = np.zeros((nsets, npairs, 2), dtype=int)
    diff_pairs_numbers_sets = np.zeros((nsets, npairs, 2), dtype=int)
    for set_cnt in range(nsets):
        same_pairs_name = []
        for pair_cnt in range(npairs):  # Same Person
            line_cnt += 1
            assert (line_cnt == 1 + set_cnt * npairs * 2 + pair_cnt)
            line_current = pairs_file[line_cnt].strip('\n').split('\t')
            assert(len(line_current)==3)
            # fname1 = '{:s}_{:04d}'.format(line_current[0], int(line_current[1]))
            # fname2 = '{:s}_{:04d}'.format(line_current[0], int(line_current[2]))
            same_pairs_name.append(line_current[0])
            same_pairs_numbers_sets[set_cnt, pair_cnt, :] = [int(line_current[1]), int(line_current[2])]
        same_pairs_name_sets.append(same_pairs_name)
        diff_pairs_names = []
        for pair_cnt in range(npairs):  # Diff Persons
            line_cnt += 1
            assert (line_cnt == 1 + set_cnt * npairs * 2 + npairs + pair_cnt)
            line_current = pairs_file[line_cnt].strip('\n').split('\t')
            assert(len(line_current)==4)
            fname1 = '{:s}_{:04d}'.format(line_current[0], int(line_current[1]))
            fname2 = '{:s}_{:04d}'.format(line_current[2], int(line_current[3]))
            diff_pairs_names.append([line_current[0], line_current[2]])
            diff_pairs_numbers_sets[set_cnt, pair_cnt, :] = [int(line_current[1]), int(line_current[3])]
        diff_pairs_names_sets.append(diff_pairs_names)
    f.close()
    np.save(mydir + 'same_pairs_name_sets.npy', same_pairs_name_sets)
    np.save(mydir + 'diff_pairs_names_sets.npy', diff_pairs_names_sets)
    np.save(mydir + 'same_pairs_numbers_sets.npy', same_pairs_numbers_sets)
    np.save(mydir + 'diff_pairs_numbers_sets.npy', diff_pairs_numbers_sets)
    pass
    return same_pairs_name_sets, diff_pairs_names_sets, same_pairs_numbers_sets, diff_pairs_numbers_sets, nsets, npairs

def check_no_overlap():
    clear_all_files = 1
    for set_cnt in range(nsets):
        for npair in range(npairs):
            if (npair + 1) % 50 == 0:
                print(set_cnt, npair + 1)
            set_fgsm_same_fname = dataset_path+ '/lfw_funneled/{:s}/same_pic{:d}_set_and_pair.txt'.format(
                same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 1])
            if clear_all_files:
                if os.path.exists(set_fgsm_same_fname):
                    os.remove(set_fgsm_same_fname)
            else:
                if os.path.exists(set_fgsm_same_fname):
                    data_from_file = np.loadtxt(set_fgsm_same_fname)
                    assert data_from_file == [set_cnt, npair]
                else:
                    np.savetxt(set_fgsm_same_fname, np.array([set_cnt, npair]), fmt='%d')

    for set_cnt in range(nsets):
        for npair in range(npairs):
            if (npair + 1) % 50 == 0:
                print(set_cnt, npair + 1)
            set_fgsm_diff_fname = dataset_path+ '/lfw_funneled/{:s}/diff_pic{:d}_set_and_pair'.format(
                diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_numbers_sets[set_cnt, npair, 1])
            if clear_all_files:
                if os.path.exists(set_fgsm_diff_fname):
                    os.remove(set_fgsm_diff_fname)
            else:
                if os.path.exists(set_fgsm_diff_fname):
                    data_from_file = np.loadtxt(set_fgsm_diff_fname)
                    assert data_from_file == [set_cnt, npair]
                else:
                    np.savetxt(set_fgsm_diff_fname, np.atleast_1d([set_cnt, npair]), fmt='%d')

def write_tensor_to_jpg_img(img_array, fname_str):
    scal_fact = int(np.floor(255 / max(img_array.reshape((-1, 1))))[0])
    np.savetxt(fname_str + '_sf.txt', np.atleast_1d(scal_fact), fmt='%d')
    img_rgb = np.moveaxis(np.uint8(np.floor(img_array * scal_fact)), 0, -1)
    img_pil = Image.fromarray(img_rgb, 'RGB')
    img_pil.save(fname_str + '.jpg')

def read_jpg_img_to_tensor(fname_str):
    img_pil = Image.open(fname_str + '.jpg')
    scal_fact = int(np.loadtxt(fname_str + '_sf.txt'))
    img_array = np.array(img_pil)
    img_array = np.moveaxis(img_array, -1, 0)
    img_array = np.float32(img_array) / scal_fact
    return torch.Tensor(img_array)

if __name__ == '__main__':
    t0 = time.time()
    print_hi('FRDM')
    lfw_ds = LFWPeople(root='./', download=True)

    (same_pairs_name_sets, diff_pairs_names_sets, same_pairs_numbers_sets, diff_pairs_numbers_sets,
     nsets, npairs) = read_pairs_lfw()
    # check_no_overlap()

    # Initialize MTCNN for face detection
    mtcnn = MTCNN()

    # Load pre-trained Inception ResNet model
    resnet = InceptionResnetV1(pretrained='casia-webface').eval()

    thresh = 1.2
    show_hist = 0
    show_hist_all_sets = 1
    rerun = [0, 0]
    # rerun = [1, 1]
    regen_fgsm = 0
    #sets_vec = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sets_vec = [0]
    # fgsm_perturb_weight_vec = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050]
    # fgsm_perturb_weight_vec = [0.002, 0.005, 0.010, 0.015, 0.020]
    fgsm_perturb_weight_vec = [0.02]
    # fgsm_perturb_weight_vec = [0.030, 0.050]
    fgsm_cnt = 0
    TP_fgsm_acc_vec = np.zeros(1 + len(fgsm_perturb_weight_vec))
    TN_fgsm_acc_vec = np.zeros(1 + len(fgsm_perturb_weight_vec))
    FP_fgsm_acc_vec = np.zeros(1 + len(fgsm_perturb_weight_vec))
    imgs_hist_same = []
    imgs_hist_diff = []
    fp_grid_imgs = []
    fp_grid_imgs_cnt = 0
    for fgsm_perturb_weight in fgsm_perturb_weight_vec:
        fgsm_cnt += 1
        same_dist_all_sets = []
        same_fgsm_dist_all_sets = []
        diff_dist_all_sets = []
        diff_fgsm_dist_all_sets = []
        for set_cnt in sets_vec:  #range(nsets):
            TP, FN, TP_fgsm, FN_fgsm, TN, FP = 0, 0, 0, 0, 0, 0
            if rerun[0] or not os.path.isfile(mydir + 'same_dist_set' + str(set_cnt) + '.npy'):
                # SAME PERSON
                same_dist = []
                same_fgsm_dist = []
                for npair in range(npairs):
                    if (npair + 1) % 50 == 0:
                        print(npair + 1)
                    image1_path = dataset_path+ '/lfw_funneled/{:s}/{:s}_{:04d}.jpg'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 0])
                    image1_cleartgt = dataset_path+ '/lfw_funneled/{:s}/{:s}_{:04d}_cleartgt'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 0])
                    image2_path = dataset_path+ '/lfw_funneled/{:s}/{:s}_{:04d}.jpg'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 1])
                    image2_fgsm_same = dataset_path+ '/lfw_funneled/{:s}/{:s}_{:04d}_fgsm{:.3f}_same_{:d}_{:d}'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 1], fgsm_perturb_weight, set_cnt, npair)
                    image2_clear = dataset_path+ '/lfw_funneled/{:s}/{:s}_{:04d}_clear'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 1])

                    img1 = Image.open(image1_path)
                    img2 = Image.open(image2_path)
                    # Detect faces and extract embeddings
                    faces1, _ = mtcnn.detect(img1)
                    faces2, _ = mtcnn.detect(img2)

                    if faces1 is not None and faces2 is not None:

                        aligned1 = mtcnn(img1)
                        aligned2 = mtcnn(img2)

                        aligned1 = make_grid(aligned1, normalize=True, value_range=(-1, 1))
                        aligned2 = make_grid(aligned2, normalize=True, value_range=(-1, 1))

                        if fgsm_cnt == 1:
                            write_tensor_to_jpg_img(aligned1, image1_cleartgt)
                            write_tensor_to_jpg_img(aligned2, image2_clear)

                        aligned2.requires_grad = True

                        embeddings1 = resnet(aligned1.unsqueeze(0)).detach()
                        embeddings2 = resnet(aligned2.unsqueeze(0))

                        # Calculate the Euclidean distance between embeddings
                        distance = (embeddings1 - embeddings2).norm().item()
                        same_dist.append(distance)

                        # Apply FGSM to SAME
                        if not os.path.exists(image2_fgsm_same+'.jpg') or regen_fgsm:
                            loss = torch.nn.MSELoss()(embeddings1, embeddings2)
                            resnet.zero_grad()
                            loss.backward()
                            perturbation = fgsm_perturb_weight * torch.sign(aligned2.grad)
                            aligned2_fgsm = (aligned2 + perturbation).detach()
                            aligned2_fgsm[aligned2_fgsm < 0] = 0
                            aligned2_fgsm[aligned2_fgsm > 1] = 1
                            write_tensor_to_jpg_img(aligned2_fgsm, image2_fgsm_same)
                        else:
                            aligned2_fgsm = read_jpg_img_to_tensor(image2_fgsm_same)

                        embeddings2_fgsm = resnet(aligned2_fgsm.unsqueeze(0)).detach()
                        aligned2_fgsm.requires_grad = False
                        distance_fgsm = (embeddings1 - embeddings2_fgsm).norm().item()
                        same_fgsm_dist.append(distance_fgsm)
                        img2_clear = aligned2
                        # save_image(img2_fgsm, image2_fgsm_same, normalize=True, value_range=(-1, 1))
                np.save(mydir + 'same_dist_set' + str(set_cnt) + '.npy', same_dist)
                np.save(mydir + 'same_fgsm{:.3f}_dist_set{:d}.npy'.format(fgsm_perturb_weight, set_cnt), same_fgsm_dist)

            if rerun[1] or not os.path.isfile(mydir + 'diff_dist_set' + str(set_cnt) + '.npy'):
                diff_dist = []
                diff_fgsm_dist = []
                # DIFF PERSON
                for npair in range(npairs):
                    if (npair + 1) % 50 == 0:
                        print(npair + 1)
                    image1_path = dataset_path+ '/lfw_funneled/{:s}/{:s}_{:04d}.jpg'.format(diff_pairs_names_sets[set_cnt][npair][0], diff_pairs_names_sets[set_cnt][npair][0], diff_pairs_numbers_sets[set_cnt, npair, 0])
                    image1_cleartgt = dataset_path+ '/lfw_funneled/{:s}/{:s}_{:04d}_cleartgt'.format(diff_pairs_names_sets[set_cnt][npair][0], diff_pairs_names_sets[set_cnt][npair][0], diff_pairs_numbers_sets[set_cnt, npair, 0])
                    image2_path = dataset_path+ '/lfw_funneled/{:s}/{:s}_{:04d}.jpg'.format(diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_numbers_sets[set_cnt, npair, 1])
                    image2_fgsm_diff = dataset_path+ '/lfw_funneled/{:s}/{:s}_{:04d}_fgsm{:.3f}_diff_{:d}_{:d}'.format(diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_numbers_sets[set_cnt, npair, 1], fgsm_perturb_weight, set_cnt, npair)
                    image2_clear = dataset_path+ '/lfw_funneled/{:s}/{:s}_{:04d}_clear'.format(diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_numbers_sets[set_cnt, npair, 1])

                    img1 = Image.open(image1_path)
                    img2 = Image.open(image2_path)
                    # Detect faces and extract embeddings
                    faces1, _ = mtcnn.detect(img1)
                    faces2, _ = mtcnn.detect(img2)

                    if faces1 is not None and faces2 is not None:
                        aligned1_temp = mtcnn(img1)
                        aligned2_temp = mtcnn(img2)

                        aligned1 = make_grid(aligned1_temp, normalize=True, value_range=(-1, 1))
                        aligned2 = make_grid(aligned2_temp, normalize=True, value_range=(-1, 1))

                        if fgsm_cnt == 1:
                            write_tensor_to_jpg_img(aligned1, image1_cleartgt)
                            write_tensor_to_jpg_img(aligned2, image2_clear)

                        img1_cleartgt = aligned1.detach()
                        img2_clear = aligned2.detach()

                        aligned2.requires_grad = True

                        embeddings1 = resnet(aligned1.unsqueeze(0)).detach()
                        embeddings2 = resnet(aligned2.unsqueeze(0))

                        # Calculate the Euclidean distance between embeddings
                        distance = (embeddings1 - embeddings2).norm().item()
                        diff_dist.append(distance)

                        # Apply FGSM to DIFF
                        if not os.path.exists(image2_fgsm_diff+'.jpg') or regen_fgsm:
                            loss = torch.nn.MSELoss()(embeddings1, embeddings2)
                            resnet.zero_grad()
                            loss.backward()
                            perturbation = fgsm_perturb_weight * torch.sign(aligned2.grad)
                            aligned2_fgsm = (aligned2 - perturbation).detach()

                            aligned2_fgsm[aligned2_fgsm < 0] = 0
                            aligned2_fgsm[aligned2_fgsm > 1] = 1
                            write_tensor_to_jpg_img(aligned2_fgsm, image2_fgsm_diff)
                        else:
                            aligned2_fgsm = read_jpg_img_to_tensor(image2_fgsm_diff)

                        embeddings2_fgsm = resnet(aligned2_fgsm.unsqueeze(0)).detach()
                        aligned2_fgsm.requires_grad = False
                        distance_fgsm = (embeddings1 - embeddings2_fgsm).norm().item()
                        diff_fgsm_dist.append(distance_fgsm)

                        if 0:  #fgsm_perturb_weight == 0.03 and distance_fgsm < 0.8:
                            print(distance_fgsm)
                            # Grid_FP = make_grid([img1_cleartgt, img2_fgsm, img2_clear], 3)
                            # fp_grid_imgs.extend([read_image(image1_cleartgt), read_image(image2_fgsm), read_image(image2_clear)])
                            fp_grid_imgs.extend([img1_cleartgt, img2_fgsm, img2_clear])
                            fp_grid_imgs_cnt += 1
                            if fp_grid_imgs_cnt == 8:
                                Grid_FP = make_grid(fp_grid_imgs,3)
                                imgs_grid_fp = torchvision.transforms.ToPILImage()(Grid_FP)
                                imgs_grid_fp.show()
                                pass

                np.save(mydir + 'diff_dist_set' + str(set_cnt) + '.npy', diff_dist)
                np.save(mydir + 'diff_fgsm{:.3f}_dist_set{:d}.npy'.format(fgsm_perturb_weight, set_cnt), diff_fgsm_dist)
            # RESULTS
            same_dist = np.load(mydir + 'same_dist_set' + str(set_cnt) + '.npy')
            same_fgsm_dist = np.load(mydir + 'same_fgsm{:.3f}_dist_set{:d}.npy'.format(fgsm_perturb_weight, set_cnt))
            diff_dist = np.load(mydir + 'diff_dist_set' + str(set_cnt) + '.npy')
            diff_fgsm_dist = np.load(mydir + 'diff_fgsm{:.3f}_dist_set{:d}.npy'.format(fgsm_perturb_weight, set_cnt))
            same_dist_all_sets.extend(same_dist)
            same_fgsm_dist_all_sets.extend(same_fgsm_dist)
            diff_dist_all_sets.extend(diff_dist)
            diff_fgsm_dist_all_sets.extend(diff_fgsm_dist)

            TP = sum(same_dist < thresh)
            FN = npairs - TP
            TP_fgsm = sum(same_fgsm_dist < thresh)
            FN_fgsm = npairs - TP_fgsm
            TN = sum(diff_dist >= thresh)
            FP = npairs - TN
            TN_fgsm = sum(diff_fgsm_dist >= thresh)
            FP_fgsm = npairs - TN_fgsm
            res_str = 'FGSM={:.3f} Set={:d} SAME:TP={:d},FN={:d}. FGSM:TP={:d},FN={:d}. DIFF:TN={:d},FP={:d}. FGSM:TN={:d},FP={:d}.'.format(fgsm_perturb_weight, set_cnt, TP, FN, TP_fgsm, FN_fgsm, TN, FP, TN_fgsm, FP_fgsm)
            if show_hist:
                res_str_accuracy = 'SAME:{:d}%, FGSM:{:d}%, DIFF:{:d}%.'.format((TP*100//300), (TP_fgsm*100//300), (TN*100//300))
                plt.hist(diff_dist, label='DIFF')
                plt.hist(same_dist, label='SAME')
                plt.hist(same_fgsm_dist, label='FGSM')
                plt.title('[Set'+str(set_cnt)+' Histogram] WPerturb='+str(fgsm_perturb_weight)+'\nThresh='+str(thresh)+': '+res_str_accuracy)
                plt.legend()
                plt.grid()
                plt.show()
            print(res_str)
        TP_acc = sum(np.array(same_dist_all_sets) < thresh) * 100 // (nsets*npairs)
        TP_fgsm_acc = sum(np.array(same_fgsm_dist_all_sets) < thresh) * 100 // (nsets*npairs)
        TN_acc = sum(np.array(diff_dist_all_sets) >= thresh) * 100 // (nsets*npairs)
        FP_acc = sum(np.array(diff_dist_all_sets) < thresh) * 100 // (nsets*npairs)
        TN_fgsm_acc = sum(np.array(diff_fgsm_dist_all_sets) >= thresh) * 100 // (nsets*npairs)
        FP_fgsm_acc = sum(np.array(diff_fgsm_dist_all_sets) < thresh) * 100 // (nsets*npairs)

        if show_hist_all_sets:
            nbins = 30
            if not os.path.isdir('plots'):
                os.makedirs('plots')
            if fgsm_cnt == 1:
                res_str_accuracy = 'SAME:{:d}%, DIFF:{:d}%.'.format(TP_acc, TN_acc)
                plt.hist(diff_dist_all_sets, nbins, label='DIFF', color='darkred')
                plt.hist(same_dist_all_sets, nbins, label='SAME', color='darkgreen')
                plt.title('[All Sets Histogram] \nThresh=' + str(thresh) + ': ' + res_str_accuracy)
                plt.legend()
                plt.grid()
                plt.savefig('./plots/hist_same_and_diff.jpg', format='jpg')
                imgs_hist_same.append(read_image('./plots/hist_same_and_diff.jpg'))
                imgs_hist_diff.append(read_image('./plots/hist_same_and_diff.jpg'))
                plt.show()

            #SAME
            res_str_accuracy = 'SAME:{:d}%, SAME-FGSM:{:d}%, DIFF:{:d}%.'.format(TP_acc, TP_fgsm_acc, TN_acc)
            plt.hist(diff_dist_all_sets, nbins, label='DIFF', color='darkred')
            plt.hist(same_dist_all_sets, nbins, label='SAME', color='darkgreen')
            plt.hist(same_fgsm_dist_all_sets, nbins, label='SAME FGSM', color='yellowgreen')
            plt.title('[All Sets Histogram] FGSMCoeff=' + str(fgsm_perturb_weight) + '\nThresh=' + str(
                thresh) + ': ' + res_str_accuracy)
            plt.legend()
            plt.grid()
            if not os.path.isdir('plots'):
                os.makedirs('plots')
            plt.savefig('./plots/hist_same_fgsm{:.3f}.jpg'.format(fgsm_perturb_weight), format='jpg')
            plt.show()
            imgs_hist_same.append(read_image('./plots/hist_same_fgsm{:.3f}.jpg'.format(fgsm_perturb_weight)))

            #DIFF
            res_str_accuracy = 'SAME:{:d}%, DIFF_FGSM:{:d}%, DIFF:{:d}%.'.format(TP_acc, TN_fgsm_acc, TN_acc)
            plt.hist(diff_dist_all_sets, nbins, label='DIFF', color='darkred')
            plt.hist(same_dist_all_sets, nbins, label='SAME', color='darkgreen')
            plt.hist(diff_fgsm_dist_all_sets, nbins, label='DIFF FGSM', color='red')
            plt.title('[All Sets Histogram] FGSMCoeff=' + str(fgsm_perturb_weight) + '\nThresh=' + str(
                thresh) + ': ' + res_str_accuracy)
            plt.legend()
            plt.grid()
            plt.savefig('./plots/hist_diff_fgsm{:.3f}.jpg'.format(fgsm_perturb_weight), format='jpg')
            plt.show()
            if not os.path.isdir('plots'):
                os.makedirs('plots')
            imgs_hist_diff.append(read_image('./plots/hist_diff_fgsm{:.3f}.jpg'.format(fgsm_perturb_weight)))

        TP_fgsm_acc_vec[fgsm_cnt] = TP_fgsm_acc
        TN_fgsm_acc_vec[fgsm_cnt] = TN_fgsm_acc
        FP_fgsm_acc_vec[fgsm_cnt] = FP_fgsm_acc
    TP_fgsm_acc_vec[0] = TP_acc
    TN_fgsm_acc_vec[0] = TN_acc
    FP_fgsm_acc_vec[0] = FP_acc
    axis_x = [0]
    axis_x.extend(fgsm_perturb_weight_vec)
    plt.plot(axis_x, TP_fgsm_acc_vec, 'og-', label='SAME')
    plt.plot(axis_x, FP_fgsm_acc_vec, 'sr-', label='DIFF')
    plt.xlabel('FGSM Coefficient')
    plt.ylabel('Authentication Rate [%]')
    plt.legend()
    plt.title('LFW FaceNet Authentication with FGSM, Thresh ='+str(thresh))
    plt.grid(True)
    plt.savefig('./plots/authentication_rate_with_fgsm.jpg', format='jpg')
    plt.show()
    print('runtime =', time.time() - t0, ' sec')

    # GRID
    set_cnt = 0
    npairs_grid = 8
    npair_vec = [100, 101, 102, 103, 106, 120, 108, 110]
    imgs_same = []
    imgs_diff = []
    for npair in npair_vec:
        image2_clear_same = dataset_path+ '/lfw_funneled/{:s}/{:s}_{:04d}_clear'.format(
            same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair],
            same_pairs_numbers_sets[set_cnt, npair, 1])
        image2_clear = read_jpg_img_to_tensor(image2_clear_same)
        imgs_same.append(image2_clear)

        image2_clear_diff =dataset_path+ '/lfw_funneled/{:s}/{:s}_{:04d}_clear'.format(
            diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_names_sets[set_cnt][npair][1],
            diff_pairs_numbers_sets[set_cnt, npair, 1])
        image2_clear = read_jpg_img_to_tensor(image2_clear_diff)
        imgs_diff.append(image2_clear)

        for fgsm_perturb_weight in fgsm_perturb_weight_vec:
            image2_fgsm_same =dataset_path+ '/lfw_funneled/{:s}/{:s}_{:04d}_fgsm{:.3f}_same_{:d}_{:d}'.format(
                same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair],
                same_pairs_numbers_sets[set_cnt, npair, 1], fgsm_perturb_weight, set_cnt, npair)
            img_fgsm = read_jpg_img_to_tensor(image2_fgsm_same)
            imgs_same.append(img_fgsm)

            image2_fgsm_diff = dataset_path+ '/lfw_funneled/{:s}/{:s}_{:04d}_fgsm{:.3f}_diff_{:d}_{:d}'.format(
                diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_names_sets[set_cnt][npair][1],
                diff_pairs_numbers_sets[set_cnt, npair, 1], fgsm_perturb_weight, set_cnt, npair)
            img_fgsm = read_jpg_img_to_tensor(image2_fgsm_diff)
            imgs_diff.append(img_fgsm)

    Grid1 = make_grid(imgs_same, 1+len(fgsm_perturb_weight_vec))
    imgs_grid1 = torchvision.transforms.ToPILImage()(Grid1)
    imgs_grid1.show()

    Grid2 = make_grid(imgs_diff, 1+len(fgsm_perturb_weight_vec))
    imgs_grid2 = torchvision.transforms.ToPILImage()(Grid2)
    imgs_grid2.show()

    imgs_hist_same.extend(imgs_hist_diff)
    Grid4 = make_grid(imgs_hist_same, 1 + len(fgsm_perturb_weight_vec))
    imgs_grid4 = torchvision.transforms.ToPILImage()(Grid4)
    imgs_grid4.show()

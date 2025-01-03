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
mydir = './data/'

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def read_pairs_lfw():
    f = open('data/pairs.txt', "r")
    pairs_file = f.readlines()
    firstline = pairs_file[0].strip('\n').split('\t')
    assert (len(firstline) == 2)
    nsets, npairs = int(firstline[0]), int(firstline[1])
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


if __name__ == '__main__':
    print_hi('FRDM')
    lfw_ds = LFWPeople(root='./', download=True)

    (same_pairs_name_sets, diff_pairs_names_sets, same_pairs_numbers_sets, diff_pairs_numbers_sets,
     nsets, npairs) = read_pairs_lfw()

    # Initialize MTCNN for face detection
    mtcnn = MTCNN()

    # Load pre-trained Inception ResNet model
    resnet = InceptionResnetV1(pretrained='casia-webface').eval()

    nsets = 10
    thresh = 1.2
    show_hist = 1
    rerun = [1, 0]
    diffpure_strength = 0.25
    # diffpure_strength = 0.05
    for set_cnt in range(10):
        TP, FN, TP_dp, FN_dp, TN, FP = 0, 0, 0, 0, 0, 0
        if rerun[0] or not os.path.isfile(mydir + 'same_dist_set' + str(set_cnt) + '.npy'):
            # SAME PERSON
            same_dist = []
            same_dist_dp = []
            for npair in range(npairs):
                if (npair + 1) % 20 == 0:
                    print(npair + 1)
                image1_path = './lfw-py/lfw_funneled/{:s}/{:s}_{:04d}.jpg'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 0])
                image2_path = './lfw-py/lfw_funneled/{:s}/{:s}_{:04d}.jpg'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 1])
                image2_path_dp = './lfw_diffpure/strgh_{:.2f}_numsteps_40/{:s}/{:s}_{:04d}.jpg'.format(diffpure_strength, same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 1])
                img1 = Image.open(image1_path)
                img2 = Image.open(image2_path)
                img2_dp = Image.open(image2_path_dp)
                # Detect faces and extract embeddings
                faces1, _ = mtcnn.detect(img1)
                faces2, _ = mtcnn.detect(img2)
                faces2_dp, _ = mtcnn.detect(img2_dp)

                if faces1 is not None and faces2 is not None and faces2_dp is not None:
                    aligned1 = mtcnn(img1).unsqueeze(0)
                    aligned2 = mtcnn(img2).unsqueeze(0)
                    aligned2_dp = mtcnn(img2_dp).unsqueeze(0)
                    embeddings1 = resnet(aligned1).detach()
                    embeddings2 = resnet(aligned2).detach()
                    embeddings2_dp = resnet(aligned2_dp).detach()

                    # Calculate the Euclidean distance between embeddings
                    distance = (embeddings1 - embeddings2).norm().item()
                    if distance < thresh:  # You can adjust the threshold for verification
                        TP += 1
                    else:
                        FN += 1
                    same_dist.append(distance)

                    distance_dp = (embeddings1 - embeddings2_dp).norm().item()
                    if distance_dp < thresh:  # You can adjust the threshold for verification
                        TP_dp += 1
                    else:
                        FN_dp += 1
                    same_dist_dp.append(distance_dp)
            np.save(mydir + 'same_dist_set' + str(set_cnt) + '.npy', same_dist)
            np.save(mydir + 'same_dist_dp{:.2f}_set{:d}.npy'.format(diffpure_strength, set_cnt), same_dist_dp)

        if rerun[1] or not os.path.isfile(mydir + 'diff_dist_set' + str(set_cnt) + '.npy'):
            diff_dist = []
            # DIFF PERSON
            for npair in range(npairs):
                if (npair + 1) % 20 == 0:
                    print(npair + 1)
                image1_path = './lfw-py/lfw_funneled/{:s}/{:s}_{:04d}.jpg'.format(diff_pairs_names_sets[set_cnt][npair][0], diff_pairs_names_sets[set_cnt][npair][0], diff_pairs_numbers_sets[set_cnt, npair, 0])
                image2_path = './lfw-py/lfw_funneled/{:s}/{:s}_{:04d}.jpg'.format(diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_numbers_sets[set_cnt, npair, 1])
                img1 = Image.open(image1_path)
                img2 = Image.open(image2_path)
                # Detect faces and extract embeddings
                faces1, _ = mtcnn.detect(img1)
                faces2, _ = mtcnn.detect(img2)

                if faces1 is not None and faces2 is not None:
                    aligned1 = mtcnn(img1).unsqueeze(0)
                    aligned2 = mtcnn(img2).unsqueeze(0)
                    embeddings1 = resnet(aligned1).detach()
                    embeddings2 = resnet(aligned2).detach()

                    # Calculate the Euclidean distance between embeddings
                    distance = (embeddings1 - embeddings2).norm().item()
                    if distance < thresh:  # You can adjust the threshold for verification
                        FP += 1
                    else:
                        TN += 1
                    diff_dist.append(distance)
            np.save(mydir + 'diff_dist_set' + str(set_cnt) + '.npy', diff_dist)

        same_dist = np.load(mydir + 'same_dist_set' + str(set_cnt) + '.npy')
        same_dist_dp = np.load(mydir + 'same_dist_dp{:.2f}_set{:d}.npy'.format(diffpure_strength, set_cnt))
        diff_dist = np.load(mydir + 'diff_dist_set' + str(set_cnt) + '.npy')
        TP = sum(same_dist < thresh)
        FN = npairs - TP
        TP_dp = sum(same_dist_dp < thresh)
        FN_dp = npairs - TP_dp
        TN = sum(diff_dist >= thresh)
        FP = npairs - TN
        res_str = 'SAME:TP={:d},FN={:d}. DIFF:TN={:d},FP={:d}.'.format(TP, FN, TN, FP)
        res_str_accuracy = 'SAME:{:d}%, SAME-DP:{:d}%, DIFF:{:d}%.'.format((TP*100//300), (TP_dp*100//300), (TN*100//300))
        if show_hist:
            plt.hist(same_dist, label='SAME')
            plt.hist(same_dist_dp, label='SAME DP[{:.2f}]'.format(diffpure_strength))
            plt.hist(diff_dist, label='DIFF')
            plt.title('[Set'+str(set_cnt)+'] Thresh='+str(thresh)+': '+res_str_accuracy)
            plt.legend()
            plt.grid()
            plt.show()
        print(res_str)

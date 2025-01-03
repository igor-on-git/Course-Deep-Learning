import numpy as np


def read_pairs_lfw(path):
    f = open(path+'/pairs.txt', "r")
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
            assert (len(line_current) == 3)
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
            assert (len(line_current) == 4)
            fname1 = '{:s}_{:04d}'.format(line_current[0], int(line_current[1]))
            fname2 = '{:s}_{:04d}'.format(line_current[2], int(line_current[3]))
            diff_pairs_names.append([line_current[0], line_current[2]])
            diff_pairs_numbers_sets[set_cnt, pair_cnt, :] = [int(line_current[1]), int(line_current[3])]
        diff_pairs_names_sets.append(diff_pairs_names)
    f.close()
    """
    np.save(mydir + 'same_pairs_name_sets.npy', same_pairs_name_sets)
    np.save(mydir + 'diff_pairs_names_sets.npy', diff_pairs_names_sets)
    np.save(mydir + 'same_pairs_numbers_sets.npy', same_pairs_numbers_sets)
    np.save(mydir + 'diff_pairs_numbers_sets.npy', diff_pairs_numbers_sets)
    pass
    """
    return same_pairs_name_sets, diff_pairs_names_sets, same_pairs_numbers_sets, diff_pairs_numbers_sets, nsets, npairs

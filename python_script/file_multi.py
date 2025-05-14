import numpy as np

from rram_fuc import *
import matplotlib.pyplot as plt

import torch

"""
1k阵列写入测试
共512个权重
权重写入128 个态 读电流-320uA ~ -5uA & 5uA ~ 320uA
"""

if __name__ == '__main__':
    print('1k_test.py')

    def expand4_matrix(X):
        # Define a function to split a number into 4 parts based on the given rules
        def split_number(W):
            parts = [0, 0, 0, 0]

            # Case 1: W < 16
            if W < 17:
                parts[3] = W  # Place W in the last position
            # Case 2: 16 < W < 33
            elif 16 < W < 33:
                parts[2] = W - 16
                parts[3] = 16
            # Case 3: W >= 33
            elif 32 < W < 49:
                parts[1] = W - 32
                parts[2] = 16
                parts[3] = 16
            else :
                parts[0] = W-48
                parts[1] = 16
                parts[2] = 16
                parts[3] = 16
            return parts

        # Create a 4x16 matrix
        Y = []

        for row in X:
            expanded_row = []
            for value in row:
                expanded_row.extend(split_number(value))  # Expand the value
            Y.append(expanded_row)

        return np.array(Y)

    def expand2_matrix(X):
        # Define a function to split a number into 4 parts based on the given rules
        def split_number(W):
            parts = [0, 0]

            # Case 1: W < 16
            if W < 33:
                parts[1] = W  # Place W in the last position
            # Case 2: 16 < W < 33
            else:
                parts[0] = W - 32
                parts[1] = 32
            return parts

        # Create a 4x16 matrix
        Y = []

        for row in X:
            expanded_row = []
            for value in row:
                expanded_row.extend(split_number(value))  # Expand the value
            Y.append(expanded_row)

        return np.array(Y)

    wrow_matrix = np.full((4,4),57)

    w4_matrix = expand4_matrix(wrow_matrix)
    w4_matrix = w4_matrix + 1
    irow4_matrix = w4_matrix*15+10

    w2_matrix = expand2_matrix(wrow_matrix)
    w2_matrix = w2_matrix + 1
    irow2_matrix = w2_matrix*7.5+17.5
    irow2_matrix = np.round(irow2_matrix, 1)

    #print(irow4_matrix)
    i_target_1k_test4, mask4 = g_matrix_padding4(irow4_matrix)

    #print(irow2_matrix)
    i_target_1k_test2, mask2 = g_matrix_padding2(irow2_matrix)

    #np.savetxt('matrix_output4.txt', i_target_1k_test4, fmt='%d')
    #np.savetxt('matrix_output2.txt', i_target_1k_test2, fmt='%d')
    '''
    # demo 3 : rram_reflect_part
    gpad_matrix, mask = g_matrix_padding2(grow_matrix)
    print(gpad_matrix)
    print(mask)

    i_target_1k_test = rram_reflect(gpad_matrix, 100)

    print(f'I min {i_target_1k_test.min()}')
    print(f'I max {i_target_1k_test.max()}')

    # demo 4 : create_forming_file
    create_forming_file('1k_test_forming')

    # demo 5 : create_write_file
    #1 9 17 25 33 41 49 57
    '''
    create_write_file('1k_4_weight57', i_target_1k_test4,mask4)
    create_write_file('1k_2_weight57', i_target_1k_test2, mask2)

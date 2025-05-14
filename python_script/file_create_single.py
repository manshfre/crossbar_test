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

    #grow_matrix = np.full((8,8),0.25)

    #grow_matrix = np.full((8, 8), 0.40)
    #grow_matrix = np.full((8, 8), 0.55)
    #grow_matrix = np.full((8, 8), 0.70)
    #grow_matrix = np.full((8, 8), 0.85)
    #grow_matrix = np.full((8, 8), 1.00)
    # grow_matrix = np.full((8, 8), 1.15)
    #grow_matrix = np.full((8, 8), 1.30)
    # grow_matrix = np.full((8, 8), 1.45)
    #grow_matrix = np.full((8, 8), 1.60)
    # grow_matrix = np.full((8, 8), 1.75)
    #grow_matrix = np.full((8, 8), 1.90)
    # grow_matrix = np.full((8, 8), 2.05)
    #grow_matrix = np.full((8, 8), 2.20)
    # grow_matrix = np.full((8, 8), 2.35)
    grow_matrix = np.full((8, 8), 2.50)
    # grow_matrix = np.full((8, 8), 2.65)

    print(grow_matrix)

    # demo 3 : rram_reflect_part
    gpad_matrix, mask = g_matrix_padding(grow_matrix)

    i_target_1k_test = rram_reflect(gpad_matrix, 100)

    print(f'I min {i_target_1k_test.min()}')
    print(f'I max {i_target_1k_test.max()}')

    # demo 4 : create_forming_file
    #create_forming_file('1k_test_forming')

    # demo 5 : create_write_file
    create_write_file('1k_test_write2500', i_target_1k_test,mask)
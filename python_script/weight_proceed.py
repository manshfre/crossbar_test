import numpy as np
from rram_fuc import *
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

if __name__ == '__main__':

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

    def analyze_matrix1(matrix, ref_mean):
        """
        """
        arr = np.array(matrix, dtype=float)
        flat = arr.flatten()

        if flat.size <= 1:
            raise ValueError("矩阵元素数量必须大于1，才能剔除最大值。")

        # 剔除最大值
        max_index = np.argmax(flat)
        reduced = np.delete(flat, max_index)

        # 均值和标准差
        mean_val = np.mean(reduced)

        # 伪标准差
        pseudo_std = np.sqrt(np.mean((reduced - ref_mean) ** 2))

        return mean_val, pseudo_std

    def cal_matrix_dropmax(irow_matrix, ird_matrix, W):
        # 将输入矩阵转换为 NumPy 数组
        irow_matrix = np.array(irow_matrix)
        ird_matrix = np.array(ird_matrix)

        # 检查输入矩阵大小是否相等
        if irow_matrix.shape != ird_matrix.shape:
            raise ValueError("输入的两个矩阵大小必须相等")

        # 沿行方向从左往右进行元素相加合并生成新矩阵
        mrow_matrix = np.add.reduceat(irow_matrix, np.arange(0, irow_matrix.shape[1], W), axis=1)
        mrd_matrix = np.add.reduceat(ird_matrix, np.arange(0, ird_matrix.shape[1], W), axis=1)

        # 对新生成的矩阵进行计算
        wrow_matrix = (mrow_matrix - W * 250) / (W * 37.5)
        wrd_matrix = (mrd_matrix - W * 250) / (W * 37.5)

        # 计算差矩阵
        mean_val,pseudo_std=analyze_matrix1(wrd_matrix,wrow_matrix[0][0])

        # 返回差矩阵的均值
        return mean_val,pseudo_std

    def cal_matrix(irow_matrix, ird_matrix, W):
        # 将输入矩阵转换为 NumPy 数组
        irow_matrix = np.array(irow_matrix)
        ird_matrix = np.array(ird_matrix)

        # 检查输入矩阵大小是否相等
        if irow_matrix.shape != ird_matrix.shape:
            raise ValueError("输入的两个矩阵大小必须相等")

        # 沿行方向从左往右进行元素相加合并生成新矩阵
        mrow_matrix = np.add.reduceat(irow_matrix, np.arange(0, irow_matrix.shape[1], W), axis=1)
        mrd_matrix = np.add.reduceat(ird_matrix, np.arange(0, ird_matrix.shape[1], W), axis=1)

        # 对新生成的矩阵进行计算
        wrow_matrix = (mrow_matrix - W * 250) / (W * 37.5)
        wrd_matrix = (mrd_matrix - W * 250) / (W * 37.5)

        # 计算差矩阵
        std_rd= np.sqrt(np.mean((wrd_matrix - wrow_matrix[0][0]) ** 2))

        # 返回差矩阵的均值
        return np.mean(wrd_matrix),std_rd


    wrow_matrix_1 = np.full((4, 4), 1)

    w4_matrix_1 = expand4_matrix(wrow_matrix_1)
    w4_matrix_1 = w4_matrix_1 + 1
    irow4_matrix_1 = w4_matrix_1*15+10

    w2_matrix_1 = expand2_matrix(wrow_matrix_1)
    w2_matrix_1 = w2_matrix_1 + 1
    irow2_matrix_1 = w2_matrix_1*7.5+17.5
    irow2_matrix_1 = np.round(irow2_matrix_1, 1)

    i4_readall_1 = np.load('../post_weight_log/2025-04-29_chip9_weight1to464.npy')
    i4_readall_1 = i4_readall_1[-1] * 1e6  # 大小为32x32

    i2_readall_1 = np.load('../post_weight_log/2025-04-29_chip9_weight1to232.npy')
    i2_readall_1 = i2_readall_1[-1] * 1e6


    ird4_matrix_1 = rram_read4(i4_readall_1, 100, irow4_matrix_1)  # 映射成4x16
    ird2_matrix_1 = rram_read2(i2_readall_1, 100, irow2_matrix_1)  # 映射成4x8


    mean_4_1,std_4_1=cal_matrix_dropmax(irow4_matrix_1*10,ird4_matrix_1*10,4)
    mean_2_1, std_2_1 = cal_matrix_dropmax(irow2_matrix_1*10, ird2_matrix_1*10, 2)
    print(
        f'mean_4_1 : {mean_4_1:.4f} std_4_1 : {std_4_1:.4f} mean_2_1 : {mean_2_1:.4f} std_2_1 : {std_2_1:.4f}')
    # -----------------------------------------------------------------------------------------------------------------------
    wrow_matrix_9 = np.full((4, 4), 9)

    w4_matrix_9 = expand4_matrix(wrow_matrix_9)
    w4_matrix_9 = w4_matrix_9 + 1
    irow4_matrix_9 = w4_matrix_9 * 15 + 10

    w2_matrix_9 = expand2_matrix(wrow_matrix_9)
    w2_matrix_9 = w2_matrix_9 + 1
    irow2_matrix_9 = w2_matrix_9 * 7.5 + 17.5
    irow2_matrix_9 = np.round(irow2_matrix_9, 1)

    i4_readall_9 = np.load('../post_weight_log/2025-04-29_chip9_weight9to464.npy')
    i4_readall_9 = i4_readall_9[-1] * 1e6  # 大小为32x32

    i2_readall_9 = np.load('../post_weight_log/2025-04-29_chip9_weight9to232.npy')
    i2_readall_9 = i2_readall_9[-1] * 1e6

    ird4_matrix_9 = rram_read4(i4_readall_9, 100, irow4_matrix_9)  # 映射成4x16
    ird2_matrix_9 = rram_read2(i2_readall_9, 100, irow2_matrix_9)  # 映射成4x8

    mean_4_9, std_4_9 = cal_matrix_dropmax(irow4_matrix_9 * 10, ird4_matrix_9 * 10, 4)
    mean_2_9, std_2_9 = cal_matrix_dropmax(irow2_matrix_9 * 10, ird2_matrix_9 * 10, 2)
    print(
        f'mean_4_9 : {mean_4_9:.4f} std_4_9 : {std_4_9:.4f} mean_2_9 : {mean_2_9:.4f} std_2_9 : {std_2_9:.4f}')
#-----------------------------------------------------------------------------------------------------------------------
    wrow_matrix_17 = np.full((4, 4), 17)

    w4_matrix_17 = expand4_matrix(wrow_matrix_17)
    w4_matrix_17 = w4_matrix_17 + 1
    irow4_matrix_17 = w4_matrix_17 * 15 + 10

    w2_matrix_17 = expand2_matrix(wrow_matrix_17)
    w2_matrix_17 = w2_matrix_17 + 1
    irow2_matrix_17 = w2_matrix_17 * 7.5 + 17.5
    irow2_matrix_17 = np.round(irow2_matrix_17, 1)

    i4_readall_17 = np.load('../post_weight_log/2025-04-29_chip9_weight17to464.npy')
    i4_readall_17 = i4_readall_17[-1] * 1e6  # 大小为 32x32

    i2_readall_17 = np.load('../post_weight_log/2025-04-29_chip9_weight17to232.npy')
    i2_readall_17 = i2_readall_17[-1] * 1e6

    ird4_matrix_17 = rram_read4(i4_readall_17, 100, irow4_matrix_17)  # 映射成 4x16
    ird2_matrix_17 = rram_read2(i2_readall_17, 100, irow2_matrix_17)  # 映射成 4x8

    mean_4_17, std_4_17 = cal_matrix_dropmax(irow4_matrix_17 * 10, ird4_matrix_17 * 10, 4)
    mean_2_17, std_2_17 = cal_matrix_dropmax(irow2_matrix_17 * 10, ird2_matrix_17 * 10, 2)
    print(
        f'mean_4_17 : {mean_4_17:.4f} std_4_17 : {std_4_17:.4f} mean_2_17 : {mean_2_17:.4f} std_2_17 : {std_2_17:.4f}')
    # -----------------------------------------------------------------------------------------------------------------------
    wrow_matrix_25 = np.full((4, 4), 25)

    w4_matrix_25 = expand4_matrix(wrow_matrix_25)
    w4_matrix_25 = w4_matrix_25 + 1
    irow4_matrix_25 = w4_matrix_25 * 15 + 10

    w2_matrix_25 = expand2_matrix(wrow_matrix_25)
    w2_matrix_25 = w2_matrix_25 + 1
    irow2_matrix_25 = w2_matrix_25 * 7.5 + 17.5
    irow2_matrix_25 = np.round(irow2_matrix_25, 1)

    i4_readall_25 = np.load('../post_weight_log/2025-04-29_chip9_weight25to464.npy')
    i4_readall_25 = i4_readall_25[-1] * 1e6  # 大小为 32x32

    i2_readall_25 = np.load('../post_weight_log/2025-04-29_chip9_weight25to232.npy')
    i2_readall_25 = i2_readall_25[-1] * 1e6

    ird4_matrix_25 = rram_read4(i4_readall_25, 100, irow4_matrix_25)  # 映射成 4x16
    ird2_matrix_25 = rram_read2(i2_readall_25, 100, irow2_matrix_25)  # 映射成 4x8

    mean_4_25, std_4_25 = cal_matrix_dropmax(irow4_matrix_25 * 10, ird4_matrix_25 * 10, 4)
    mean_2_25, std_2_25 = cal_matrix_dropmax(irow2_matrix_25 * 10, ird2_matrix_25 * 10, 2)
    print(
        f'mean_4_25 : {mean_4_25:.4f} std_4_25 : {std_4_25:.4f} mean_2_25 : {mean_2_25:.4f} std_2_25 : {std_2_25:.4f}')
    # -----------------------------------------------------------------------------------------------------------------------
    wrow_matrix_33 = np.full((4, 4), 33)

    w4_matrix_33 = expand4_matrix(wrow_matrix_33)
    w4_matrix_33 = w4_matrix_33 + 1
    irow4_matrix_33 = w4_matrix_33 * 15 + 10

    w2_matrix_33 = expand2_matrix(wrow_matrix_33)
    w2_matrix_33 = w2_matrix_33 + 1
    irow2_matrix_33 = w2_matrix_33 * 7.5 + 17.5
    irow2_matrix_33 = np.round(irow2_matrix_33, 1)

    i4_readall_33 = np.load('../post_weight_log/2025-04-29_chip9_weight33to464.npy')
    i4_readall_33 = i4_readall_33[-1] * 1e6  # 大小为 32x32

    i2_readall_33 = np.load('../post_weight_log/2025-04-29_chip9_weight33to232.npy')
    i2_readall_33 = i2_readall_33[-1] * 1e6

    ird4_matrix_33 = rram_read4(i4_readall_33, 100, irow4_matrix_33)  # 映射成 4x16
    ird2_matrix_33 = rram_read2(i2_readall_33, 100, irow2_matrix_33)  # 映射成 4x8

    mean_4_33, std_4_33 = cal_matrix_dropmax(irow4_matrix_33 * 10, ird4_matrix_33 * 10, 4)
    mean_2_33, std_2_33 = cal_matrix_dropmax(irow2_matrix_33 * 10, ird2_matrix_33 * 10, 2)
    print(
        f'mean_4_33 : {mean_4_33:.4f} std_4_33 : {std_4_33:.4f} mean_2_33 : {mean_2_33:.4f} std_2_33 : {std_2_33:.4f}')
    # -----------------------------------------------------------------------------------------------------------------------
    wrow_matrix_41 = np.full((4, 4), 41)

    w4_matrix_41 = expand4_matrix(wrow_matrix_41)
    w4_matrix_41 = w4_matrix_41 + 1
    irow4_matrix_41 = w4_matrix_41 * 15 + 10

    w2_matrix_41 = expand2_matrix(wrow_matrix_41)
    w2_matrix_41 = w2_matrix_41 + 1
    irow2_matrix_41 = w2_matrix_41 * 7.5 + 17.5
    irow2_matrix_41 = np.round(irow2_matrix_41, 1)

    i4_readall_41 = np.load('../post_weight_log/2025-04-29_chip9_weight41to464.npy')
    i4_readall_41 = i4_readall_41[-1] * 1e6  # 大小为 32x32

    i2_readall_41 = np.load('../post_weight_log/2025-04-29_chip9_weight41to232.npy')
    i2_readall_41 = i2_readall_41[-1] * 1e6

    ird4_matrix_41 = rram_read4(i4_readall_41, 100, irow4_matrix_41)  # 映射成 4x16
    ird2_matrix_41 = rram_read2(i2_readall_41, 100, irow2_matrix_41)  # 映射成 4x8

    mean_4_41, std_4_41 = cal_matrix_dropmax(irow4_matrix_41 * 10, ird4_matrix_41 * 10, 4)
    mean_2_41, std_2_41 = cal_matrix_dropmax(irow2_matrix_41 * 10, ird2_matrix_41 * 10, 2)
    print(
        f'mean_4_41 : {mean_4_41:.4f} std_4_41 : {std_4_41:.4f} mean_2_41 : {mean_2_41:.4f} std_2_41 : {std_2_41:.4f}')
    # -----------------------------------------------------------------------------------------------------------------------

    wrow_matrix_49 = np.full((4, 4), 49)

    w4_matrix_49 = expand4_matrix(wrow_matrix_49)
    w4_matrix_49 = w4_matrix_49 + 1
    irow4_matrix_49 = w4_matrix_49 * 15 + 10

    w2_matrix_49 = expand2_matrix(wrow_matrix_49)
    w2_matrix_49 = w2_matrix_49 + 1
    irow2_matrix_49 = w2_matrix_49 * 7.5 + 17.5
    irow2_matrix_49 = np.round(irow2_matrix_49, 1)

    i4_readall_49 = np.load('../post_weight_log/2025-04-29_chip9_weight49to464.npy')
    i4_readall_49 = i4_readall_49[-1] * 1e6  # 大小为 32x32

    i2_readall_49 = np.load('../post_weight_log/2025-04-29_chip9_weight49to232.npy')
    i2_readall_49 = i2_readall_49[-1] * 1e6

    ird4_matrix_49 = rram_read4(i4_readall_49, 100, irow4_matrix_49)  # 映射成 4x16
    ird2_matrix_49 = rram_read2(i2_readall_49, 100, irow2_matrix_49)  # 映射成 4x8

    mean_4_49, std_4_49 = cal_matrix_dropmax(irow4_matrix_49 * 10, ird4_matrix_49 * 10, 4)
    mean_2_49, std_2_49 = cal_matrix_dropmax(irow2_matrix_49 * 10, ird2_matrix_49 * 10, 2)
    print(
        f'mean_4_49 : {mean_4_49:.4f} std_4_49 : {std_4_49:.4f} mean_2_49 : {mean_2_49:.4f} std_2_49 : {std_2_49:.4f}')

    # -----------------------------------------------------------------------------------------------------------------------
    wrow_matrix_57 = np.full((4, 4), 57)

    w4_matrix_57 = expand4_matrix(wrow_matrix_57)
    w4_matrix_57 = w4_matrix_57 + 1
    irow4_matrix_57 = w4_matrix_57 * 15 + 10

    w2_matrix_57 = expand2_matrix(wrow_matrix_57)
    w2_matrix_57 = w2_matrix_57 + 1
    irow2_matrix_57 = w2_matrix_57 * 7.5 + 17.5
    irow2_matrix_57 = np.round(irow2_matrix_57, 1)

    i4_readall_57 = np.load('../post_weight_log/2025-04-29_chip9_weight57to464.npy')
    i4_readall_57 = i4_readall_57[-1] * 1e6  # 大小为 32x32

    i2_readall_57 = np.load('../post_weight_log/2025-04-29_chip9_weight57to232.npy')
    i2_readall_57 = i2_readall_57[-1] * 1e6

    ird4_matrix_57 = rram_read4(i4_readall_57, 100, irow4_matrix_57)  # 映射成 4x16
    ird2_matrix_57 = rram_read2(i2_readall_57, 100, irow2_matrix_57)  # 映射成 4x8

    mean_4_57, std_4_57 = cal_matrix_dropmax(irow4_matrix_57 * 10, ird4_matrix_57 * 10, 4)
    mean_2_57, std_2_57 = cal_matrix_dropmax(irow2_matrix_57 * 10, ird2_matrix_57 * 10, 2)
    print(
        f'mean_4_57 : {mean_4_57:.4f} std_4_57 : {std_4_57:.4f} mean_2_57 : {mean_2_57:.4f} std_2_57 : {std_2_57:.4f}')
    # -----------------------------------------------------------------------------------------------------------------------

    std_4_list = [std_4_1,std_4_9,std_4_17,std_4_25,std_4_33,std_4_41,std_4_49,std_4_57]
    std_2_list = [std_2_1,std_2_9,std_2_17,std_2_25,std_2_33,std_2_41,std_2_49,std_2_57]

    x_ticks = [1, 9, 17, 25, 33, 41, 49, 57]

    plt.figure(figsize=(8, 5))
    plt.plot(x_ticks, std_2_list, marker='o', label='code by 2')
    plt.plot(x_ticks, std_4_list, marker='s', label='code by 4')

    plt.xlabel('Weight_ideal', fontsize=14)
    plt.ylabel('Weight_std', fontsize=14)
    plt.title('map variation', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)  # 图例放在右下角
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    mean_4_list = [mean_4_1-1,mean_4_9-9,mean_4_17-17,mean_4_25-25,mean_4_33-33,mean_4_41-41,mean_4_49-49,mean_4_57-57]
    mean_2_list = [mean_2_1-1,mean_2_9-9,mean_2_17-17,mean_2_25-25,mean_2_33-33,mean_2_41-41,mean_2_49-49,mean_2_57-57]

    x_ticks = [1, 9, 17, 25, 33, 41, 49, 57]

    plt.figure(figsize=(8, 5))
    plt.plot(x_ticks, mean_2_list, marker='o', label='code by 2')
    plt.plot(x_ticks, mean_4_list, marker='s', label='code by 4')

    plt.xlabel('Weight_ideal', fontsize=14)
    plt.ylabel('Weight_mean', fontsize=14)
    plt.title('map equivalent', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)  # 图例放在右下角
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    '''
    grd_list = [754,727,716,719,758,801,874,962]
    grdEXP_list = [707,709,662,621,616,630,727,760]

    x_ticks2 = [400,700,1000,1300,1600,1900,2200,2500]
    plt.figure(figsize=(8, 5))
    plt.plot(x_ticks2, grd_list, marker='o', label='origin')
    plt.plot(x_ticks2, grdEXP_list, marker='s', label='expedite')

    plt.xlabel('Conductance [μS]', fontsize=14)
    plt.ylabel('Time [S]', fontsize=14)
    plt.title('Time versus', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)  # 图例放在右下角
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    '''
import numpy as np
from rram_fuc import*
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

if __name__ == '__main__':

    grow_matrix_16 = np.full((8,8),2.50) #单位ms

    i_readall_16 = np.load('../post_log/2025-04-29_chip9_level16to88.npy')
    i_readall_16 = i_readall_16[-1] * 1e6 #大小为32x32

    iEXP_readall_16 = np.load('../post_log/2025-04-29_chip9_level16to88EXP.npy')
    iEXP_readall_16 = iEXP_readall_16[-1] * 1e6

    i_process_16= np.load('../post_log/2025-04-29_chip9_level1688opi.npy')
    v_process_16= np.load('../post_log/2025-04-29_chip9_level1688vol.npy')

    iEXP_process_16= np.load('../post_log/2025-04-29_chip9_level1688EXPopi.npy')
    vEXP_process_16= np.load('../post_log/2025-04-29_chip9_level1688EXPvol.npy')

    def extract_g_over(A: np.ndarray) -> np.ndarray:
        """

        """
        if A.shape[0] != 64:
            raise ValueError("输入矩阵的行数必须为 64")

        rev_idx = np.argmax(A[:, ::-1] != 0, axis=1)
        last_idx = A.shape[1] - 1 - rev_idx
        last_vals = A[np.arange(A.shape[0]), last_idx]
        return last_vals.reshape((8, 8)) * 10

    g_over_16 = extract_g_over(i_process_16)#操作完成电导，单位us
    '''
    index = np.argmax(g_over)
    # 将一维索引转换为二维坐标
    row1, col1 = np.unravel_index(index, g_over.shape)
    print(f'{row1} {col1}')
    '''
    grd_matrix_16 = rram_read(i_readall_16, 100, grow_matrix_16)#映射成8x8
    grdEXP_matrix_16 = rram_read(iEXP_readall_16,100,grow_matrix_16)

    '''
    index = np.argmax(grd_matrix)
    # 将一维索引转换为二维坐标
    row2, col2 = np.unravel_index(index, grd_matrix.shape)
    print(f'{row2} {col2}')
    '''

    def analyze_matrix(matrix, ref_mean):
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
        std_val = np.std(reduced)

        # 伪标准差
        pseudo_std = np.sqrt(np.mean((reduced - ref_mean) ** 2))

        return mean_val, std_val, pseudo_std

    mean_rd_16, std_rd_16, pseudo_std_rd_16 = analyze_matrix(grd_matrix_16 * 1e3, 2500)
    print(f'mean_rd_16 : {mean_rd_16:.4f} std_rd : {std_rd_16:.4f} pseudo_std_rd_16: {pseudo_std_rd_16:.4f}')

    mean_rdEXP_16, std_rdEXP_16, pseudo_std_rdEXP_16 = analyze_matrix(grdEXP_matrix_16 * 1e3, 2500)
    print(f'mean_rdEXP_16 : {mean_rdEXP_16:.4f} std_rdEXP_16 : {std_rdEXP_16:.4f} pseudo_std_rdEXP_16: {pseudo_std_rdEXP_16:.4f}')


    def compare_process_matrices(i_process, iEXP_process):
        """

        """
        # 转为 NumPy 数组
        A = np.array(i_process, dtype=float)
        B = np.array(iEXP_process, dtype=float)
        if A.shape != B.shape:
            raise ValueError("两个输入矩阵必须具有相同的形状")

        # 每行非零元素计数
        number1 = np.count_nonzero(A, axis=1)
        number2 = np.count_nonzero(B, axis=1)

        # 全矩阵非零元素总数
        total1 = int(np.count_nonzero(A))
        total2 = int(np.count_nonzero(B))

        # 计算 versus 列向量
        # 1 if number1 >= number2 else 0
        versus = (number1 >= number2).astype(int).reshape(-1, 1)

        # ratio_EXP：判断 1 的数量是否多于 0
        ones = int(np.sum(versus == 1))
        zeros = int(np.sum(versus == 0))
        ratio_EXP = 100*ones/(zeros+ones)

        # totalgap 与 total_EXP
        totalgap = total1 - total2
        total_EXP = totalgap > 0

        return versus, totalgap, ratio_EXP, total_EXP

    versus_16, totalgap_16, ratio_EXP_16, total_EXP_16 = compare_process_matrices(i_process_16, iEXP_process_16)
    print(f'versus_16 : {versus_16} totalgap_16: {totalgap_16} ratio_EXP_16 : {ratio_EXP_16} total_EXP_16: {total_EXP_16}')
# -----------------------------------------------------------------------------------------------------------------------
    grow_matrix_14 = np.full((8, 8), 2.20)  # 单位 ms

    # 加载并处理读出电流矩阵
    i_readall_14 = np.load('../post_log/2025-04-29_chip9_level14to88.npy')
    i_readall_14 = i_readall_14[-1] * 1e6  # 大小为 32x32

    iEXP_readall_14 = np.load('../post_log/2025-04-29_chip9_level14to88EXP.npy')
    iEXP_readall_14 = iEXP_readall_14[-1] * 1e6

    # 加载并处理过程矩阵
    i_process_14 = np.load('../post_log/2025-04-29_chip9_level1488opi.npy')
    v_process_14 = np.load('../post_log/2025-04-29_chip9_level1488vol.npy')

    iEXP_process_14 = np.load('../post_log/2025-04-29_chip9_level1488EXPopi.npy')
    vEXP_process_14 = np.load('../post_log/2025-04-29_chip9_level1488EXPvol.npy')


    # 计算电导矩阵
    g_over_14 = extract_g_over(i_process_14)  # 操作完成电导，单位 µs

    # 映射读出电流到 8x8 网格
    grd_matrix_14 = rram_read(i_readall_14, 100, grow_matrix_14)
    grdEXP_matrix_14 = rram_read(iEXP_readall_14, 100, grow_matrix_14)

    # 统计并打印读出电阻统计量
    mean_rd_14, std_rd_14, pseudo_std_rd_14 = analyze_matrix(grd_matrix_14 * 1e3, 2200)
    print(
        f'mean_rd_14      : {mean_rd_14:.4f} \nstd_rd_14       : {std_rd_14:.4f} \npseudo_std_rd_14: {pseudo_std_rd_14:.4f}')

    mean_rdEXP_14, std_rdEXP_14, pseudo_std_rdEXP_14 = analyze_matrix(grdEXP_matrix_14 * 1e3, 2200)
    print(
        f'mean_rdEXP_14      : {mean_rdEXP_14:.4f} \nstd_rdEXP_14       : {std_rdEXP_14:.4f} \npseudo_std_rdEXP_14: {pseudo_std_rdEXP_14:.4f}')


    # 对比过程矩阵并打印结果
    versus_14, totalgap_14, ratio_EXP_14, total_EXP_14 = compare_process_matrices(
        i_process_14, iEXP_process_14
    )
    print(
        f'versus_14   : {versus_14} \ntotalgap_14 : {totalgap_14} \nratio_EXP_14: {ratio_EXP_14} \ntotal_EXP_14: {total_EXP_14}')

#-----------------------------------------------------------------------------------------------------------------------
    grow_matrix_12 = np.full((8, 8), 1.90)  # 单位 ms

    # 加载并处理读出电流矩阵
    i_readall_12 = np.load('../post_log/2025-04-29_chip9_level12to88.npy')
    i_readall_12 = i_readall_12[-1] * 1e6  # 大小为 32x32

    iEXP_readall_12 = np.load('../post_log/2025-04-29_chip9_level12to88EXP.npy')
    iEXP_readall_12 = iEXP_readall_12[-1] * 1e6

    # 加载并处理过程矩阵
    i_process_12 = np.load('../post_log/2025-04-29_chip9_level1288opi.npy')
    v_process_12 = np.load('../post_log/2025-04-29_chip9_level1288vol.npy')

    iEXP_process_12 = np.load('../post_log/2025-04-29_chip9_level1288EXPopi.npy')
    vEXP_process_12 = np.load('../post_log/2025-04-29_chip9_level1288EXPvol.npy')


    # 计算电导矩阵
    g_over_12 = extract_g_over(i_process_12)  # 操作完成电导，单位 µs

    # 映射读出电流到 8x8 网格
    grd_matrix_12 = rram_read(i_readall_12, 100, grow_matrix_12)
    grdEXP_matrix_12 = rram_read(iEXP_readall_12, 100, grow_matrix_12)


    # 统计并打印读出电阻统计量
    mean_rd_12, std_rd_12, pseudo_std_rd_12 = analyze_matrix(grd_matrix_12 * 1e3, 1900)
    print(
        f'mean_rd_12      : {mean_rd_12:.4f} \nstd_rd_12       : {std_rd_12:.4f} \npseudo_std_rd_12: {pseudo_std_rd_12:.4f}')

    mean_rdEXP_12, std_rdEXP_12, pseudo_std_rdEXP_12 = analyze_matrix(grdEXP_matrix_12 * 1e3, 1900)
    print(
        f'mean_rdEXP_12      : {mean_rdEXP_12:.4f} \nstd_rdEXP_12       : {std_rdEXP_12:.4f} \npseudo_std_rdEXP_12: {pseudo_std_rdEXP_12:.4f}')


    # 对比过程矩阵并打印结果
    versus_12, totalgap_12, ratio_EXP_12, total_EXP_12 = compare_process_matrices(
        i_process_12, iEXP_process_12
    )
    print(
        f'versus_12   : {versus_12} \ntotalgap_12 : {totalgap_12} \nratio_EXP_12: {ratio_EXP_12} \ntotal_EXP_12: {total_EXP_12}')


# -----------------------------------------------------------------------------------------------------------------------
    grow_matrix_10 = np.full((8, 8), 1.60)  # 单位 ms

    # 加载并处理读出电流矩阵
    i_readall_10 = np.load('../post_log/2025-04-29_chip9_level10to88.npy')
    i_readall_10 = i_readall_10[-1] * 1e6  # 大小为 32x32

    iEXP_readall_10 = np.load('../post_log/2025-04-29_chip9_level10to88EXP.npy')
    iEXP_readall_10 = iEXP_readall_10[-1] * 1e6

    # 加载并处理过程矩阵
    i_process_10 = np.load('../post_log/2025-04-29_chip9_level1088opi.npy')
    v_process_10 = np.load('../post_log/2025-04-29_chip9_level1088vol.npy')

    iEXP_process_10 = np.load('../post_log/2025-04-29_chip9_level1088EXPopi.npy')
    vEXP_process_10 = np.load('../post_log/2025-04-29_chip9_level1088EXPvol.npy')

    # 计算电导矩阵
    g_over_10 = extract_g_over(i_process_10)  # 操作完成电导，单位 µs

    # 映射读出电流到 8x8 网格
    grd_matrix_10 = rram_read(i_readall_10, 100, grow_matrix_10)
    grdEXP_matrix_10 = rram_read(iEXP_readall_10, 100, grow_matrix_10)

    # 统计并打印读出电阻统计量
    mean_rd_10, std_rd_10, pseudo_std_rd_10 = analyze_matrix(grd_matrix_10 * 1e3, 1600)
    print(
        f'mean_rd_10      : {mean_rd_10:.4f} \nstd_rd_10       : {std_rd_10:.4f} \npseudo_std_rd_10: {pseudo_std_rd_10:.4f}')

    mean_rdEXP_10, std_rdEXP_10, pseudo_std_rdEXP_10 = analyze_matrix(grdEXP_matrix_10 * 1e3, 1600)
    print(
        f'mean_rdEXP_10      : {mean_rdEXP_10:.4f} \nstd_rdEXP_10       : {std_rdEXP_10:.4f} \npseudo_std_rdEXP_10: {pseudo_std_rdEXP_10:.4f}')

    # 对比过程矩阵并打印结果
    versus_10, totalgap_10, ratio_EXP_10, total_EXP_10 = compare_process_matrices(
        i_process_10, iEXP_process_10
    )
    print(
        f'versus_10   : {versus_10} \ntotalgap_10 : {totalgap_10} \nratio_EXP_10: {ratio_EXP_10} \ntotal_EXP_10: {total_EXP_10}')


# -----------------------------------------------------------------------------------------------------------------------
    grow_matrix_8 = np.full((8, 8), 1.30)  # 单位 ms

    # 加载并处理读出电流矩阵
    i_readall_8 = np.load('../post_log/2025-04-29_chip9_level8to88.npy')
    i_readall_8 = i_readall_8[-1] * 1e6  # 大小为 32x32

    iEXP_readall_8 = np.load('../post_log/2025-04-29_chip9_level8to88EXP.npy')
    iEXP_readall_8 = iEXP_readall_8[-1] * 1e6

    # 加载并处理过程矩阵
    i_process_8 = np.load('../post_log/2025-04-29_chip9_level888opi.npy')
    v_process_8 = np.load('../post_log/2025-04-29_chip9_level888vol.npy')

    iEXP_process_8 = np.load('../post_log/2025-04-29_chip9_level888EXPopi.npy')
    vEXP_process_8 = np.load('../post_log/2025-04-29_chip9_level888EXPvol.npy')


    # 计算电导矩阵
    g_over_8 = extract_g_over(i_process_8)  # 操作完成电导，单位 µs

    # 映射读出电流到 8x8 网格
    grd_matrix_8 = rram_read(i_readall_8, 100, grow_matrix_8)
    grdEXP_matrix_8 = rram_read(iEXP_readall_8, 100, grow_matrix_8)


    # 统计并打印读出电阻统计量
    mean_rd_8, std_rd_8, pseudo_std_rd_8 = analyze_matrix(grd_matrix_8 * 1e3, 1300)
    print(
        f'mean_rd_8      : {mean_rd_8:.4f} \nstd_rd_8       : {std_rd_8:.4f} \npseudo_std_rd_8: {pseudo_std_rd_8:.4f}')

    mean_rdEXP_8, std_rdEXP_8, pseudo_std_rdEXP_8 = analyze_matrix(grdEXP_matrix_8 * 1e3, 1300)
    print(
        f'mean_rdEXP_8      : {mean_rdEXP_8:.4f} \nstd_rdEXP_8       : {std_rdEXP_8:.4f} \npseudo_std_rdEXP_8: {pseudo_std_rdEXP_8:.4f}')


    # 对比过程矩阵并打印结果
    versus_8, totalgap_8, ratio_EXP_8, total_EXP_8 = compare_process_matrices(
        i_process_8, iEXP_process_8
    )
    print(
        f'versus_8   : {versus_8} \ntotalgap_8 : {totalgap_8} \nratio_EXP_8: {ratio_EXP_8} \ntotal_EXP_8: {total_EXP_8}')


# -----------------------------------------------------------------------------------------------------------------------
    grow_matrix_6 = np.full((8, 8), 1.00)  # 单位 ms

    # 加载并处理读出电流矩阵
    i_readall_6 = np.load('../post_log/2025-04-29_chip9_level6to88.npy')
    i_readall_6 = i_readall_6[-1] * 1e6  # 大小为 32x32

    iEXP_readall_6 = np.load('../post_log/2025-04-29_chip9_level6to88EXP.npy')
    iEXP_readall_6 = iEXP_readall_6[-1] * 1e6

    # 加载并处理过程矩阵
    i_process_6 = np.load('../post_log/2025-04-29_chip9_level688opi.npy')
    v_process_6 = np.load('../post_log/2025-04-29_chip9_level688vol.npy')

    iEXP_process_6 = np.load('../post_log/2025-04-29_chip9_level688EXPopi.npy')
    vEXP_process_6 = np.load('../post_log/2025-04-29_chip9_level688EXPvol.npy')

    # 计算电导矩阵
    g_over_6 = extract_g_over(i_process_6)  # 操作完成电导，单位 µs

    # 映射读出电流到 8x8 网格
    grd_matrix_6 = rram_read(i_readall_6, 100, grow_matrix_6)
    grdEXP_matrix_6 = rram_read(iEXP_readall_6, 100, grow_matrix_6)

    # 统计并打印读出电阻统计量
    mean_rd_6, std_rd_6, pseudo_std_rd_6 = analyze_matrix(grd_matrix_6 * 1e3, 1000)
    print(
        f'mean_rd_6      : {mean_rd_6:.4f} \nstd_rd_6       : {std_rd_6:.4f} \npseudo_std_rd_6: {pseudo_std_rd_6:.4f}')

    mean_rdEXP_6, std_rdEXP_6, pseudo_std_rdEXP_6 = analyze_matrix(grdEXP_matrix_6 * 1e3, 1000)
    print(
        f'mean_rdEXP_6      : {mean_rdEXP_6:.4f} \nstd_rdEXP_6       : {std_rdEXP_6:.4f} \npseudo_std_rdEXP_6: {pseudo_std_rdEXP_6:.4f}')

    # 对比过程矩阵并打印结果
    versus_6, totalgap_6, ratio_EXP_6, total_EXP_6 = compare_process_matrices(
        i_process_6, iEXP_process_6
    )
    print(
        f'versus_6   : {versus_6} \ntotalgap_6 : {totalgap_6} \nratio_EXP_6: {ratio_EXP_6} \ntotal_EXP_6: {total_EXP_6}')


# -----------------------------------------------------------------------------------------------------------------------
    grow_matrix_4 = np.full((8, 8), 0.70)  # 单位 ms

    # 加载并处理读出电流矩阵
    i_readall_4 = np.load('../post_log/2025-04-29_chip9_level4to88.npy')
    i_readall_4 = i_readall_4[-1] * 1e6  # 大小为 32x32

    iEXP_readall_4 = np.load('../post_log/2025-04-29_chip9_level4to88EXP.npy')
    iEXP_readall_4 = iEXP_readall_4[-1] * 1e6

    # 加载并处理过程矩阵
    i_process_4 = np.load('../post_log/2025-04-29_chip9_level488opi.npy')
    v_process_4 = np.load('../post_log/2025-04-29_chip9_level488vol.npy')

    iEXP_process_4 = np.load('../post_log/2025-04-29_chip9_level488EXPopi.npy')
    vEXP_process_4 = np.load('../post_log/2025-04-29_chip9_level488EXPvol.npy')


    # 计算电导矩阵
    g_over_4 = extract_g_over(i_process_4)  # 操作完成电导，单位 µs

    # 映射读出电流到 8x8 网格
    grd_matrix_4 = rram_read(i_readall_4, 100, grow_matrix_4)
    grdEXP_matrix_4 = rram_read(iEXP_readall_4, 100, grow_matrix_4)

    # 统计并打印读出电阻统计量
    mean_rd_4, std_rd_4, pseudo_std_rd_4 = analyze_matrix(grd_matrix_4 * 1e3, 700)
    print(
        f'mean_rd_4      : {mean_rd_4:.4f} \nstd_rd_4       : {std_rd_4:.4f} \npseudo_std_rd_4: {pseudo_std_rd_4:.4f}')

    mean_rdEXP_4, std_rdEXP_4, pseudo_std_rdEXP_4 = analyze_matrix(grdEXP_matrix_4 * 1e3, 700)
    print(
        f'mean_rdEXP_4      : {mean_rdEXP_4:.4f} \nstd_rdEXP_4       : {std_rdEXP_4:.4f} \npseudo_std_rdEXP_4: {pseudo_std_rdEXP_4:.4f}')

    # 对比过程矩阵并打印结果
    versus_4, totalgap_4, ratio_EXP_4, total_EXP_4 = compare_process_matrices(
        i_process_4, iEXP_process_4
    )
    print(
        f'versus_4   : {versus_4} \ntotalgap_4 : {totalgap_4} \nratio_EXP_4: {ratio_EXP_4} \ntotal_EXP_4: {total_EXP_4}')


# -----------------------------------------------------------------------------------------------------------------------
    grow_matrix_2 = np.full((8, 8), 0.40)  # 单位 ms

    # 加载并处理读出电流矩阵
    i_readall_2 = np.load('../post_log/2025-04-29_chip9_level2to88.npy')
    i_readall_2 = i_readall_2[-1] * 1e6  # 大小为 32x32

    iEXP_readall_2 = np.load('../post_log/2025-04-29_chip9_level2to88EXP.npy')
    iEXP_readall_2 = iEXP_readall_2[-1] * 1e6

    # 加载并处理过程矩阵
    i_process_2 = np.load('../post_log/2025-04-29_chip9_level288opi.npy')
    v_process_2 = np.load('../post_log/2025-04-29_chip9_level288vol.npy')

    iEXP_process_2 = np.load('../post_log/2025-04-29_chip9_level288EXPopi.npy')
    vEXP_process_2 = np.load('../post_log/2025-04-29_chip9_level288EXPvol.npy')

    # 计算电导矩阵
    g_over_2 = extract_g_over(i_process_2)  # 操作完成电导，单位 µs

    # 映射读出电流到 8x8 网格
    grd_matrix_2 = rram_read(i_readall_2, 100, grow_matrix_2)
    grdEXP_matrix_2 = rram_read(iEXP_readall_2, 100, grow_matrix_2)


    # 统计并打印读出电阻统计量
    mean_rd_2, std_rd_2, pseudo_std_rd_2 = analyze_matrix(grd_matrix_2 * 1e3, 400)
    print(
        f'mean_rd_2      : {mean_rd_2:.4f} \nstd_rd_2       : {std_rd_2:.4f} \npseudo_std_rd_2: {pseudo_std_rd_2:.4f}')

    mean_rdEXP_2, std_rdEXP_2, pseudo_std_rdEXP_2 = analyze_matrix(grdEXP_matrix_2 * 1e3, 400)
    print(
        f'mean_rdEXP_2      : {mean_rdEXP_2:.4f} \nstd_rdEXP_2       : {std_rdEXP_2:.4f} \npseudo_std_rdEXP_2: {pseudo_std_rdEXP_2:.4f}')


    # 对比过程矩阵并打印结果
    versus_2, totalgap_2, ratio_EXP_2, total_EXP_2 = compare_process_matrices(
        i_process_2, iEXP_process_2
    )
    print(
        f'versus_2   : {versus_2} \ntotalgap_2 : {totalgap_2} \nratio_EXP_2: {ratio_EXP_2} \ntotal_EXP_2: {total_EXP_2}')

    pseudo_std_rdEXP = [pseudo_std_rdEXP_2,pseudo_std_rdEXP_4,pseudo_std_rdEXP_6,pseudo_std_rdEXP_8,pseudo_std_rdEXP_10,pseudo_std_rdEXP_12,pseudo_std_rdEXP_14,pseudo_std_rdEXP_16]
    pseudo_std_rd = [pseudo_std_rd_2,pseudo_std_rd_4,pseudo_std_rd_6,pseudo_std_rd_8,pseudo_std_rd_10,pseudo_std_rd_12,pseudo_std_rd_14,pseudo_std_rd_16]

    x_ticks = [400, 700, 1000, 1300, 1600, 1900, 2200, 2500]

    plt.figure(figsize=(8, 5))
    plt.plot(x_ticks, pseudo_std_rdEXP, marker='o', label='pseudo_std_rdEXP')
    plt.plot(x_ticks, pseudo_std_rd, marker='s', label='pseudo_std_rd')

    plt.xlabel('Conductance [μS]',fontsize=14)
    plt.ylabel('Pseudo Std Deviation [μS]',fontsize=14)
    plt.title('Pseudo Std Deviation versus',fontsize=16)
    plt.legend(loc='upper right',fontsize=12)  # 图例放在右下角
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    totalgap = [totalgap_2,totalgap_4,totalgap_6,totalgap_8,totalgap_10,totalgap_12,totalgap_14,totalgap_16]

    # 横轴标签
    x_ticks = [400, 700, 1000, 1300, 1600, 1900, 2200, 2500]

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(x_ticks, totalgap, marker='o', color='tab:blue', linewidth=2)

    plt.xlabel('Conductance [μS]', fontsize=14)
    plt.ylabel('Totalgap', fontsize=14)
    plt.title('Pulse decreased', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    ratio_EXP = [ratio_EXP_2,ratio_EXP_4,ratio_EXP_6,ratio_EXP_8,ratio_EXP_10,ratio_EXP_12,ratio_EXP_14,ratio_EXP_16]

    # 横轴标签
    x_ticks = [400, 700, 1000, 1300, 1600, 1900, 2200, 2500]

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(x_ticks, ratio_EXP, marker='o', color='tab:blue', linewidth=2)

    plt.xlabel('Conductance [μS]', fontsize=14)
    plt.ylabel('percent', fontsize=14)
    plt.title('ratio decreased', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------计算CDF------------------------------------------------------------------
    def draw_CDF(A: np.ndarray, B: np.ndarray, C: np.ndarray = None, X=None, labels=None):
        def cdf(matrix):
            sorted_vals = np.sort(matrix.flatten())
            x = sorted_vals
            y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            return x, y

        def custom_transform(y):
            target_values = np.array([0.01, 0.1, 0.5, 0.9, 0.95, 0.99])
            target_positions = np.linspace(0, 1, len(target_values))
            return np.interp(y, target_values, target_positions)

        plt.figure(figsize=(10, 6))

        # 设置默认标签
        default_labels = ['A', 'B', 'C']
        if labels is None:
            labels = default_labels

        # 绘制 A
        x_a, y_a = cdf(A)
        plt.plot(x_a, custom_transform(y_a), 'r+', label=labels[0])

        # 绘制 B
        x_b, y_b = cdf(B)
        plt.plot(x_b, custom_transform(y_b), 'bs', label=labels[1])

        # 如果有 C
        if C is not None:
            x_c, y_c = cdf(C)
            plt.plot(x_c, custom_transform(y_c), 'g^', label=labels[2] if len(labels) > 2 else default_labels[2])

        # 绘制竖轴线
        if X is not None:
            if not isinstance(X, (list, np.ndarray)):
                X = [X]
            for x_val in X:
                plt.axvline(x=x_val, color='k', linestyle='--')

        plt.xlabel('Conductance [μS]')
        plt.ylabel('CDF')
        plt.xlim(0, 3000)
        plt.ylim(0, 1)
        plt.legend(loc='lower right')

        y_ticks = [0.01, 0.1, 0.5, 0.9, 0.95, 0.99]
        y_tick_positions = custom_transform(y_ticks)
        plt.yticks(y_tick_positions, y_ticks)

        plt.grid(True, which="major", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.show()


    def draw_CDF_multiple(i_read_list, i_over_list, X_list, read_label='i_read', over_label='i_over'):
        assert len(i_read_list) == len(i_over_list) == len(X_list), "i_read、i_over和X长度必须一致"
        N = len(i_read_list)

        plt.figure(figsize=(10, 6))

        def cdf(matrix):
            sorted_vals = np.sort(matrix.flatten())
            x = sorted_vals
            y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            return x, y

        def custom_transform(y):
            target_values = np.array([0.01, 0.1, 0.5, 0.9, 0.95, 0.99])
            target_positions = np.linspace(0, 1, len(target_values))
            return np.interp(y, target_values, target_positions)

        first_read = True
        first_over = True

        # 循环绘制每组数据
        for idx in range(N):
            A = i_read_list[idx]
            B = i_over_list[idx]
            X_val = X_list[idx]

            # 计算CDF
            x_a, y_a = cdf(A)
            x_b, y_b = cdf(B)

            # 绘制i_read曲线
            '''
            plt.plot(x_a, custom_transform(y_a), 'r+',
                     label=read_label if first_read else None)
            first_read = False  # 之后不再标label
            '''
            # 绘制i_over曲线

            plt.plot(x_b, custom_transform(y_b), 'bs',
                     label=over_label if first_over else None)
            first_over = False

            # 绘制X位置的竖线
            plt.axvline(x=X_val, color='k', linestyle='--', alpha=0.3)

        # 设置统一属性
        plt.xlabel('Conductance [μS]')
        plt.ylabel('CDF')
        plt.xlim(0, 3000)
        plt.ylim(0, 1)

        y_ticks = [0.01, 0.1, 0.5, 0.9, 0.95, 0.99]
        y_tick_positions = custom_transform(y_ticks)
        plt.yticks(y_tick_positions, y_ticks)

        plt.grid(True, which="major", ls="--", alpha=0.5)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()


    i_read_list = [grdEXP_matrix_2 * 1e3, grdEXP_matrix_4 * 1e3, grdEXP_matrix_6 * 1e3, grdEXP_matrix_8 * 1e3,grdEXP_matrix_10 * 1e3, grdEXP_matrix_12 * 1e3, grdEXP_matrix_14 * 1e3, grdEXP_matrix_16 * 1e3]
    i_over_list = [grd_matrix_2 * 1e3, grd_matrix_4 * 1e3, grd_matrix_6 * 1e3, grd_matrix_8 * 1e3,grd_matrix_10 * 1e3, grd_matrix_12 * 1e3, grd_matrix_14 * 1e3, grd_matrix_16 * 1e3]
    X_list = [400,700,1000,1300,1600, 1900, 2200, 2500]

    #draw_CDF(grdEXP_matrix * 1e3,grd_matrix * 1e3,X=2500,labels=['grdEXP_matrix','grd_matrix'])
    draw_CDF_multiple(i_read_list, i_over_list, X_list, read_label='grdEXP_matrix', over_label='grd_matrix')

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------绘制G-V图---------------------------------------------------------
    def plot_GV_on_ax(ax, vol, opi, Y, color='b', alpha=0.6):
        """ 在已有 axes 上画一组 GV 曲线和一条横线 """
        mask = vol != 0
        ax.plot(vol[mask], opi[mask], color=color, alpha=alpha)
        ax.axhline(Y, color=color, linestyle='--', alpha=0.6)

    def plot1_GV_on_ax(ax, vol, opi, Y, **kwargs):
        """ 在已有 axes 上画一组 GV 曲线和一条横线 """
        mask = vol != 0
        ax.plot(vol[mask], opi[mask], alpha=kwargs.get('alpha', 0.6))
        ax.axhline(Y, color='r', linestyle='--', alpha=0.7)


    def draw_GV(vol_array_list, opi_array_list, Y):
        """ 原 draw_GV，不变 """
        fig, ax = plt.subplots(figsize=(10, 6))
        for v, o in zip(vol_array_list, opi_array_list):
            plot1_GV_on_ax(ax, v, o, Y)
        ax.set_xlabel('V_G (V)', fontsize=14)
        ax.set_ylabel('Conductance (μS)', fontsize=14)
        ax.text(0.01, 0.98,
                "Set: V_BL=1.0V, V_SL=0V\nWL PW=100ns\nVstep=1mV|10mV\ngap=250μS",
                transform=ax.transAxes,
                fontsize=10, va='top', ha='left')
        ax.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def overlay_all(vol_array_list_list, opi_array_list_list, Y_list):
        """
        将多组 (vol_list, opi_list, Y) 叠加绘制在一张图中。
        每组使用唯一颜色，其64条曲线共享此颜色。
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # 可选颜色列表（可根据组数扩展）
        color_list = plt.cm.tab10.colors

        N = len(vol_array_list_list)
        for idx in range(N):
            vols = vol_array_list_list[idx]
            opis = opi_array_list_list[idx]
            Y = Y_list[idx]
            color = color_list[idx % len(color_list)]  # 循环使用颜色

            for v, o in zip(vols, opis):
                plot_GV_on_ax(ax, v, o, Y, color=color, alpha=0.5)

        # 设置坐标轴标签
        ax.set_xlabel('V_G (V)', fontsize=18)
        ax.set_ylabel('Conductance (μS)', fontsize=18)

        # 设置左上角文本（使用 LaTeX 语法显示 μ）
        text_content = (
                "Set: V_BL = 1.0V, V_SL = 0V\n"
                "WL PW = 100ns\n"
                "Vstep = 1mV | 10mV\n"
                "gap = 250" + r"$\mu$" + "S"
        )
        ax.text(0.01, 0.98, text_content,
                transform=ax.transAxes,
                fontsize=18,
                va='top', ha='left')

        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    #draw_GV(vEXP_process, iEXP_process * 10, Y=2500)
    #draw_GV(v_process, i_process * 10, Y=2500)
    vEXP_process_list = [vEXP_process_2,vEXP_process_4,vEXP_process_6,vEXP_process_8,vEXP_process_10,vEXP_process_12,vEXP_process_14,vEXP_process_16]
    iEXP_process_list = [iEXP_process_2*10,iEXP_process_4*10,iEXP_process_6*10,iEXP_process_8*10,iEXP_process_10*10,iEXP_process_12*10,iEXP_process_14*10,iEXP_process_16*10]
    Y_list = [400,700,1000,1300,1600, 1900, 2200, 2500]

    v_process_list = [v_process_2,v_process_4,v_process_6,v_process_8,v_process_10,v_process_12,v_process_14,v_process_16]
    i_process_list = [i_process_2*10,i_process_4*10,i_process_6*10,i_process_8*10,i_process_10*10,i_process_12*10,i_process_14*10,i_process_16*10]

    overlay_all(vEXP_process_list, iEXP_process_list, Y_list)
    overlay_all(v_process_list, i_process_list, Y_list)
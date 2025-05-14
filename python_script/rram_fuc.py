import numpy as np
import json
import pandas as pd
from datetime import datetime

func_name_forming_couple = 'FORMING_SINGLE'
func_name_write_couple = 'WRITE_SINGLE'


def weight_reflect(weight: np.ndarray, r_sample: float, v_dac_ref: float, diff_scale: float = 4.0) -> np.ndarray:
    """
    将权重矩阵转换为电导矩阵。

    :param weight: 权重
    :type weight: numpy.ndarray
    :param r_sample: 采样电阻，单位千欧
    :param v_dac_ref: DAC参考电压
    :param diff_scale: 差分电路缩放系数
    :return: 电导矩阵，单位 mS。
    """

    v_adc_ref = 3.0

    scale = v_adc_ref / (r_sample * v_dac_ref * diff_scale)
    g_matrix = weight * scale

    return g_matrix


def weight_reflect_T(g_matrix: np.ndarray, r_sample: float, v_dac_ref: float, diff_scale: float = 4.0) -> np.ndarray:
    """
    将电导矩阵还原成权重。

    :param g_matrix: 电导矩阵，单位 mS
    :type g_matrix: numpy.ndarray
    :param r_sample: 采样电阻，单位千欧
    :param v_dac_ref: DAC参考电压
    :param diff_scale: 差分电路缩放系数
    :return: 权重
    """

    v_adc_ref = 3.0

    scale = v_adc_ref / (r_sample * v_dac_ref * diff_scale)
    weight = g_matrix / scale

    return weight


def g_matrix_padding(grow_matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    # 电导矩阵扩充
    gpad_matrix = np.zeros((32, 32), dtype=float)
    mask = np.zeros((32, 32), dtype=int)
    h, w = grow_matrix.shape

    for r in range(0, h):
        for c in range(0, w):
            gpad_matrix[4 * r][4 * c] = grow_matrix[r][c]
            mask[4 * r][4 * c] = 1

    return gpad_matrix, mask

def g_matrix_padding4(grow_matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    # 电导矩阵扩充2
    gpad_matrix = np.zeros((32, 32), dtype=float)
    mask = np.zeros((32, 32), dtype=int)

    for r in range(0, 4):
        for c in range(0, 4):
            gpad_matrix[8 * r +2][8 * c + 1] = grow_matrix[r][4*c]
            gpad_matrix[8 * r + 2][8 * c + 2] = grow_matrix[r][4*c+1]
            gpad_matrix[8 * r + 2][8 * c + 3] = grow_matrix[r][4*c+2]
            gpad_matrix[8 * r + 2][8 * c + 4] = grow_matrix[r][4*c+3]

            mask[8 * r + 2][8 * c + 1] = 1
            mask[8 * r + 2][8 * c + 2] = 1
            mask[8 * r + 2][8 * c + 3] = 1
            mask[8 * r + 2][8 * c + 4] = 1

    return gpad_matrix, mask

def g_matrix_padding2(grow_matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    # 电导矩阵扩充2
    gpad_matrix = np.zeros((32, 32), dtype=float)
    mask = np.zeros((32, 32), dtype=int)

    for r in range(0, 4):
        for c in range(0, 4):
            gpad_matrix[8 * r +2][8 * c + 3] = grow_matrix[r][2*c]
            gpad_matrix[8 * r + 2][8 * c + 4] = grow_matrix[r][2*c+1]


            mask[8 * r + 2][8 * c + 3] = 1
            mask[8 * r + 2][8 * c + 4] = 1

    return gpad_matrix, mask


def rram_reflect(gpad_matrix: np.ndarray, v_read: float) -> np.ndarray:
    """
    将电导矩阵转换为目标写电流，用于T5830的写入。

    """

    print('>>> rram_reflect')

    i_reflect = np.zeros((32, 32), dtype=float)

    index = 0

    # 坐标重映射 + 参数转换
    for r in range(0, 32):
        for c in range(0, 32):
            i_reflect[r][c] = gpad_matrix[r][c] * v_read

            print(f'{index:>4} : [{r:02} {c:02}] : {i_reflect[r][c]:>7.4g} uA')
            index += 1

    print('<<< rram_reflect\n')

    return i_reflect


def create_write_file(file_name: str, i_reflect: np.ndarray, mask=None) -> None:
    """
    生成T5830差分对写入代码。

    :param file_name: 输出文件名
    :param i_reflect: 目标电导
    :param mask: 掩模，省略表示全部允许
    :return: None
    """

    print('>>> create_write_file')

    creation_time = datetime.now()
    formatted_date = creation_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f'Files are created on {formatted_date}')

    if mask is None:
        mask = np.ones((32, 32), dtype=int)

    with open(f'../loadfile/{file_name}_s.txt', 'w') as file:
        print('; >>>>>> WRITE SINGLE', file=file)
        print(f'; Created on {formatted_date}', file=file)
        for r in range(0, 32):
            for c in range(0, 32):
                if mask[r][c]:
                    print(f'GOSUB {func_name_write_couple}({r}, {c}, {i_reflect[r][c]:>.4g}UA)', file=file)
        print('; <<<<<< WRITE SINGLE', file=file)

    print('<<< create_write_file\n')


def create_forming_file(file_name: str, mask=None) -> None:
    """
    生成T5830差分对Forming代码。

    :param file_name: 输出文件名
    :param mask: 掩模，省略表示全部允许
    :return: None
    """
    print('>>> create_forming_file')

    creation_time = datetime.now()
    formatted_date = creation_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f'Files are created on {formatted_date}')

    if mask is None:
        mask = np.ones((32, 32), dtype=int)

    with open(f'../loadfile/{file_name}_s.txt', 'w') as file:
        print('; >>>>>> FORMING SINGLE', file=file)
        print(f'; Created on {formatted_date}', file=file)
        for r in range(0, 32):
            for c in range(0, 32):
                if mask[r][c]:
                    print(f'GOSUB {func_name_forming_couple}({r}, {c})', file=file)
        print('; <<<<<< FORMING SINGLE', file=file)

    print('<<< create_forming_file\n')


def rram_read(i_read: np.ndarray, v_read: float, grow_matrix: np.ndarray) -> np.ndarray:
    """
    读取T5830的扫描日志给出的读电流信息，将其转换成原电导矩阵。

    :param i_read: numpy张量，读电流（映射后），单位 uA
    :param v_read: T5830的读电压，用来计算目标电流，单位 mV
    :return: numpy张量，还原后的电导矩阵，单位 mS
    """

    print('>>> rram_reflect')

    grd_matrix = np.zeros_like(grow_matrix, dtype=float)
    h, w = grow_matrix.shape
    index = 0

    # 坐标重映射 + 参数转换
    for r in range(0, h):
        for c in range(0, w):

            grd_matrix[r][c] = i_read[4 * r][4 * c] / v_read
            g_read = grd_matrix[r][c]
            if g_read > 0.001 or g_read < -0.001:
                print(f'{index:>4} : [{r:02} {c:02}] : {g_read:>7.4g} mS')
            else:
                print(f'{index:>4} : [{r:02} {c:02}] :     < 1 uS')

            index += 1

    print('<<< rram_reflect\n')

    return grd_matrix


def rram_read2(i_read: np.ndarray, v_read: float, grow_matrix: np.ndarray) -> np.ndarray:
    """
    读取T5830的扫描日志给出的读电流信息，将其转换成原电导矩阵。

    :param i_read: numpy张量，读电流（映射后），单位 uA
    :param v_read: T5830的读电压，用来计算目标电流，单位 mV
    :return: numpy张量，还原后的电导矩阵，单位 mS
    """

    print('>>> rram_reflect')

    grd_matrix = np.zeros_like(grow_matrix, dtype=float)
    index = 0

    # 坐标重映射 + 参数转换
    for r in range(0, 4):
        for c in range(0, 4):
            grd_matrix[r][2 * c] = i_read[8 * r + 2][8 * c + 3] / v_read
            grd_matrix[r][2 * c + 1] = i_read[8 * r + 2][8 * c + 4] / v_read

            index += 1

    print('<<< rram_reflect\n')

    return grd_matrix * 100

def rram_read4(i_read: np.ndarray, v_read: float, grow_matrix: np.ndarray) -> np.ndarray:
    """
    读取T5830的扫描日志给出的读电流信息，将其转换成原电导矩阵。

    :param i_read: numpy张量，读电流（映射后），单位 uA
    :param v_read: T5830的读电压，用来计算目标电流，单位 mV
    :return: numpy张量，还原后的电导矩阵，单位 mS
    """

    print('>>> rram_reflect')

    grd_matrix = np.zeros_like(grow_matrix, dtype=float)
    index = 0

    # 坐标重映射 + 参数转换
    for r in range(0, 4):
        for c in range(0, 4):
            grd_matrix[r][4 * c] = i_read[8 * r + 2][8*c+1] / v_read
            grd_matrix[r][4 * c+1] = i_read[8 * r + 2][8 * c + 2] / v_read
            grd_matrix[r][4 * c+2] = i_read[8 * r + 2][8 * c + 3] / v_read
            grd_matrix[r][4 * c+3] = i_read[8 * r + 2][8 * c + 4] / v_read

            index += 1

    print('<<< rram_reflect\n')

    return grd_matrix *100


def g_diff_eval(grd_matrix: np.ndarray, grow_matrix: np.ndarray) -> (float, float):
    """
    比较评估读取到的导矩阵与原矩阵，计算均值与标准差。

    :param g_matrix_read: 读取到的电导矩阵 32x32
    :param g_matrix_source: 原电导矩阵
    :param x: 原电导矩阵起始行
    :param y: 原电导矩阵起始列
    :return: 误差均值、误差标准差
    """

    g_error = grd_matrix.flatten() - grow_matrix.flatten()

    return np.mean(g_error), np.std(g_error)


if __name__ == '__main__':
    print('The following are demos.\n')



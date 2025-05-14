import numpy as np

# 定义文件路径
log_file_path = '../weight_log/log2025-04-29_chip9_weight33to232.txt'
output_file_path = '../post_weight_log/2025-04-29_chip9_weight33to232.npy'
output_vol_path = '../post_weight_log/2025-04-29_chip9_weight33to232vol.npy'
output_opi_path = '../post_weight_log/2025-04-29_chip9_weight33to232opi.npy'

# 初始化变量
data_array = np.zeros((32, 32), dtype=float)
data_array_list = []

vol_array = np.zeros(801, dtype=float)#单位V
vol_array_list = []#记载64器件set过程电压

opi_array = np.zeros(801, dtype=float)#单位uA
opi_array_list = []#记载64器件set过程电流

recording = False

# 状态
STATUS_INIT = 'INIT'
STATUS_SCAN = 'SACN'
STATUS_SET = 'SET'
STATUS_RESET = 'RESET'
STATUS_FORMING = 'FORMING'

status = STATUS_INIT


def analysis_iread(str_iread):
    # 解析电流值并转换为浮点数，考虑单位
    value_iread = 0.0

    if 'NA' in str_iread:
        value_iread = float(str_iread.replace('NA', '')) * 1e-9
    elif 'UA' in str_iread:
        value_iread = float(str_iread.replace('UA', '')) * 1e-6
    elif 'MA' in str_iread:
        value_iread = float(str_iread.replace('MA', '')) * 1e-3
    elif 'A' in str_iread:
        value_iread = float(str_iread.replace('A', ''))

    return value_iread

def analysis_vread(str_vread):
    # 解析电流值并转换为浮点数，考虑单位
    value_vread = 0.0

    if 'MV' in str_vread:
        value_vread = float(str_vread.replace('MV', '')) * 1e-3
    elif 'V' in str_vread:
        value_vread = float(str_vread.replace('V', ''))

    return value_vread

row = 0
col = 0

# 打开并读取日志文件
with open(log_file_path, 'r') as file:
    for line in file:

        if '[Info]' not in line:
            continue

        # SCAN模式
        if '[SCAN_START]' in line:
            status = STATUS_SCAN
            print('[SCAN]')
            continue
        elif '[SCAN_COMPLETED]' in line:
            status = STATUS_INIT
            data_array_list.append(data_array)
            data_array = np.zeros((32, 32), dtype=float)
            continue

        # set模式
        elif '[SET_START]' in line:
            status = STATUS_SET
            INDEX = 0
            print('[SET]')
            continue
        elif '[SET_COMPLETED]' in line:
            status = STATUS_INIT

            INDEX = 0

            vol_array_list.append(vol_array)
            vol_array = np.zeros(801, dtype=float)

            opi_array_list.append(opi_array)
            opi_array = np.zeros(801, dtype=float)
            continue

        # 提取电流数据
        if status is STATUS_SCAN and 'IRD_SL' in line:
            parts = line.split()

            row = int(parts[-3])
            col = int(parts[-2])

            # 提取电流部分，假设电流数据在最后一个元素
            current_str = parts[-1]

            data_array[row][col] = analysis_iread(current_str)

        elif status is STATUS_SET and 'IRD_SL' in line:
            parts = line.split()
                # 提取电流部分，假设电流数据在最后一个元素
            current_str = parts[-1]
            vol_str = parts[-2]

            opi_array[INDEX] = analysis_iread(current_str) * 1e6
            vol_array[INDEX] = analysis_vread(vol_str)

            INDEX+=1

# 将数据保存为npy格式
np.save(output_file_path, np.array(data_array_list))
np.save(output_opi_path, np.array(opi_array_list))
np.save(output_vol_path, np.array(vol_array_list))

print("Data saved as :", output_file_path)

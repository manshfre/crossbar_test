# crossbar_test
device_code:assembly code executed by the used test device

loadfile: the txt related to memristors to be operated,should be copied to the correct part of the device code

log: document about the read current under distinct two write schemes

post_log: numpy array filled by extracted data from the log

weight_log: document about the read current under distinct two weight encoding schemes

post_weight_log: numpy array filled by extracted data from the weight_log

python_script:
file_create_single: create loadfile used in write scheme testment

file_multi: create loadfile used in weight encoding scheme testment

log_analysis: extract useful data from raw log and then create post files

rram_fuc: some functions that are called in other pyhton files

conduc_proceed: process the data from post files that origin from write scheme log

weight_proceed: process the data from post files that origin from weight encoding scheme log



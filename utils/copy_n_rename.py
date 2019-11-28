import os
from shutil import copyfile

train_path_0 = '/mnt/wrk/dataset/sec_num_prod/sec_num_prod_0/sec_nums/'
train_path_1 = '/mnt/wrk/dataset/sec_num_prod/sec_num_prod_0/sec_nums_1/'
train_path_2 = '/mnt/wrk/dataset/sec_num_prod/sec_num_prod_0/sec_nums_2/'
val_path =     '/mnt/wrk/dataset/sec_num_prod/sec_num_prod_0/sec_nums_3/'

target_train = '/mnt/wrk/dataset/sec_num_prod/train_sv/'
target_val = '/mnt/wrk/dataset/sec_num_prod/val_sv/'

k = 0
train_dirs = [train_path_0, train_path_1, train_path_2]
val_dirs = [val_path]

for i in train_dirs:
    for j in os.listdir(i + 'img/'):
        copyfile(i + 'img/' + j, target_train + 'img/a' + str(k) + '.png')
        copyfile(i + 'ann/' + j + '.json', target_train + 'ann/a' + str(k) + '.png.json')
        k += 1

k = 0
for i in val_dirs:
    for j in os.listdir(i + 'img/'):
        copyfile(i + 'img/' + j, target_val + 'img/a' + str(k) + '.png')
        copyfile(i + 'ann/' + j + '.json', target_val + 'ann/a' + str(k) + '.png.json')
        k += 1

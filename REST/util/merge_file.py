import os

source_dir = '/home/szcyxy/下载/others'
target_dir='/home/szcyxy/SSD/SimpleClickRes/model_zero_conv/evaluation_logs/others'
for dir in os.listdir(source_dir):
    source_file = os.path.join(source_dir, dir,'result_5.csv')
    target_file = os.path.join(target_dir, dir,'result_5.csv')

    f1 = open(source_file,'r')
    f2 = open(target_file,'a')
    f2.write('\n')
    f2.write(f1.read())
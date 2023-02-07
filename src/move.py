import os
from glob import glob

train_size = 0.75
for dir in glob('data/*'):
    if not os.path.exists(dir + '/train'):
        os.makedirs(dir + '/train')
        os.makedirs(dir + '/test')
    files = glob(dir + '/*.png')
    for file in files[:int(len(files)*train_size/2)]:
        os.rename(file, dir + '/train/' + os.path.basename(file))
    for file in files[int(len(files)*train_size/2):int(len(files)/2)]:
        os.rename(file, dir + '/test/' + os.path.basename(file))
    for file in files[int(len(files)/2):int(len(files)*train_size/2 + len(files)/2)]:
        os.rename(file, dir + '/train/' + os.path.basename(file))
    for file in files[int(len(files)*train_size/2 + len(files)/2):]:
        os.rename(file, dir + '/test/' + os.path.basename(file))
        
# # Path: src\move.py
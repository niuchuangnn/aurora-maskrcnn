import shutil
import os

source_folder = '/home/ljm/NiuChuang/AuroraObjectData/img/'
names_file = '/home/ljm/NiuChuang/AuroraObjectData/test_spread_416_d60s.txt'
save_folder = '/home/ljm/NiuChuang/AuroraObjectData/img_test/'
img_type = '.jpg'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

f = open(names_file, 'r')
lines = f.readlines()
num_img = len(lines)

for i in range(num_img):
    img_file = source_folder + lines[i][0:-1] + img_type
    save_file = save_folder + lines[i][0:-1] + img_type
    shutil.copy(img_file, save_file)

import numpy as np
from scipy.misc import imread

imgFolder = '/home/ljm/NiuChuang/AuroraObjectData/img/'
imgNames = '/home/ljm/NiuChuang/AuroraObjectData/images.txt'
img_type = '.jpg'

f = open(imgNames, 'r')
lines = f.readlines()
num_img = len(lines)
mean_sum = 0
for i in range(num_img):
    name = lines[i][0:-1]
    img_path = imgFolder + name + img_type

    im = imread(img_path)

    im_mean = im.mean()

    mean_sum += im_mean

mean_all = mean_sum / float(num_img)

print('mean_all', mean_all)
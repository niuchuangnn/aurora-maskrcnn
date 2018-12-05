import sys
sys.path.insert(0, '../')
from maskrcnn_benchmark.config_aurora import cfg
from predictor import AuroraDemo
import cv2
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image
import numpy as np
import math
import os

save_reults_folder = './results/'
if not os.path.exists(save_reults_folder):
    os.mkdir(save_reults_folder)

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2_aurora_width.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

aurora_demo = AuroraDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
)

imgFolder = '/home/ljm/NiuChuang/AuroraObjectData/labeled2003_38044/'
imgNames = '/home/ljm/NiuChuang/AuroraObjectData/Alllabel2003_38044_arc.txt'
img_type = '.bmp'

f = open(imgNames, 'r')
lines = f.readlines()
num_img = len(lines)
mean_sum = 0
fig_id = 1
names = []
zenith_angles = []
arc_widths = []

show_prediction = False
show_details = False
one_fig = False
mode = 'line'

img_height = 440
img_width = 440

for i in range(0, num_img):
    print(i)
    name = lines[i][0:-1]
    img_path = imgFolder + name + img_type

    image_ori = cv2.imread(img_path)

    angle = aurora_demo.compute_angle(image_ori, thresh_bdry_number=150)

    if angle is None:
        continue

    image_ori = Image.fromarray(image_ori)

    image = image_ori.rotate(angle)

    image = np.asarray(image)

    angle1 = aurora_demo.compute_angle(image, thresh_bdry_number=150)

    if angle1 is None:
        continue

    angle = angle + angle1
    image_r = image_ori.rotate(angle)

    image = np.asarray(image_r)

    if mode == 'line':
        if one_fig:
            plt.figure(fig_id)
            fig_id += 1
        zenith_angles_i, arc_widths_i = aurora_demo.compute_arc_zangle_width_intensity_line(image, show_details=show_details, one_fig=one_fig)

    if mode == 'seg':
        zenith_angles_i, arc_widths_i = aurora_demo.compute_arc_zangle_width(image, show_details=False)

    zenith_angles += zenith_angles_i
    arc_widths += arc_widths_i
    names.append(name)

    if show_prediction:
        prediction = aurora_demo.run_on_opencv_image(image, angle=-angle)
        plt.figure(fig_id)
        fig_id += 1
        plt.imshow(image_ori)
        plt.axis('off')

        line_v = np.array([[0, 219.5], [439, 219.5]])
        sin_a = math.sin(-angle*math.pi/180.)
        cos_a = math.cos(-angle*math.pi/180.)
        mtx_r = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        center_r = np.array([219.5, 219.5])
        line_r = np.dot(line_v-center_r, mtx_r) + center_r
        plt.plot(line_r[:, 1], line_r[:, 0], color='red')
        plt.scatter([219.5], [219.5], s=40, c='r')

        plt.figure(fig_id)
        fig_id += 1
        plt.imshow(prediction)
        plt.axis('off')

        plt.show()

save_zangle_width_file = '/home/ljm/NiuChuang/AuroraObjectData/zangle_width/agw_tr1058_te38044_arc_cnd2_' + mode + '.txt'
f = open(save_zangle_width_file, 'w')
for a in range(len(zenith_angles)):
    f.write(str(zenith_angles[a]) + ' ' + str(arc_widths[a]) + '\n')

plt.figure(fig_id)
fig_id += 1
plt.scatter(zenith_angles, arc_widths, s=2)
plt.show()
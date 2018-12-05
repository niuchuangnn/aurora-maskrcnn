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
import shutil

from pycocotools.coco import COCO
annFile = '/home/ljm/NiuChuang/AuroraObjectData/annotations_rect_spread_test_spread_416_d60s.json'
aurora_coco = COCO(annFile)
imgIds = aurora_coco.getImgIds()

# save_reults_folder = './results/driftArcs/'
# if not os.path.exists(save_reults_folder):
#     os.mkdir(save_reults_folder)

config_file = "./configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2_aurora.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

aurora_demo = AuroraDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
)

# load image and then run prediction
# imgFolder = '/home/ljm/NiuChuang/AuroraObjectData/img/'
# imgNames = '/home/ljm/NiuChuang/AuroraObjectData/images.txt'
# img_type = '.jpg'
imgFolder = '/home/ljm/NiuChuang/AuroraObjectData/labeled2003_38044/'
# imgNames = '/home/ljm/NiuChuang/AuroraObjectData/Alllabel2003_38044_arc.txt'
imgNames = '/home/ljm/NiuChuang/AuroraObjectData/test_spread_416_d60s.txt'
img_type = '.bmp'

f = open(imgNames, 'r')
lines = f.readlines()
num_img = len(lines)
mean_sum = 0
fig_id = 1
names = []
zenith_angles = []
arc_widths = []

show_prediction = True
show_one_stage = True
mode = 'line'

img_height = 440
img_width = 440

img_folder = '/home/ljm/NiuChuang/DriftArcs/'
files = os.listdir(img_folder)
point_names = [fn[0:-4] for fn in files]

# point_names = ['N20031221G121911', 'N20031222G031921', 'N20031222G051821',
#                'N20031223G062721', 'N20031223G140901', 'N20040101G134942',
#                'N20040116G050623', 'N20040117G062333', 'N20040118G051051',
#                'N20040118G071552']

# for i in range(17, num_img):
#     print(i)
#     name = lines[i][0:-1]
#
#     # Specify name.
#     name = 'N20040101G134942'
for name in point_names:

    img_path = imgFolder + name + img_type

    image_ori = cv2.imread(img_path)

    # # Save figure of original image.
    # fig = plt.figure(fig_id)
    # fig_id += 1
    # plt.imshow(image_ori, cmap='gray')
    # plt.axis('off')
    # aurora_demo.save_figure(fig, img_height, img_width, name, save_folder=save_reults_folder)

    # shutil.copy(img_path, save_reults_folder+name+img_type)

    # if name in imgIds:
    #     # Save label figure.
    #     # assert name in imgIds
    #     annIds = aurora_coco.getAnnIds(imgIds=[name])
    #     anns = aurora_coco.loadAnns(annIds)
    #     fig = plt.figure(fig_id)
    #     fig_id += 1
    #     plt.imshow(image_ori, cmap='gray')
    #     plt.axis('off')
    #     aurora_coco.showAnns(anns)
    #     ax = plt.gca()
    #
    #     for b in range(len(anns)):
    #         ann = anns[b]
    #         bbox = ann['bbox']
    #
    #         xmin = bbox[0]
    #         ymin = bbox[1]
    #         bbox_weight = bbox[2]
    #         bbox_height = bbox[3]
    #
    #         coords = (xmin, ymin), bbox_weight, bbox_height
    #         ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='g', linewidth=3))
    #
    #         ax.text(xmin, ymin, 'Arc',
    #                 bbox={'facecolor': 'g', 'alpha': 0.5})
    #     aurora_demo.save_figure(fig, img_height, img_width, name + "_label", save_folder=save_reults_folder)

    # # Save prediction figure of the one-stage process.
    # prediction = aurora_demo.run_on_opencv_image(image_ori, angle=0)
    # fig = plt.figure(fig_id)
    # fig_id += 1
    # plt.imshow(prediction)
    # plt.axis('off')
    # aurora_demo.save_figure(fig, img_height, img_width, name+"_one_stage", save_folder=save_reults_folder)

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

    # # Save the intensity-zangle curve.
    # fig = plt.figure(fig_id)
    # fig_id += 1
    # zenith_angles_i, arc_widths_i = aurora_demo.compute_arc_zangle_width_intensity_line(image, show_details=True, one_fig=True)
    # plt.savefig(save_reults_folder+name+'_intensity.png', dpi=300, bbox_inches='tight', transparent=True)
    #
    # zenith_angles += zenith_angles_i
    # arc_widths += arc_widths_i
    # names.append(name)

    # # Save prediction figure of the two-stage process without rotation.
    # prediction = aurora_demo.run_on_opencv_image(image, angle=0)
    # fig = plt.figure(fig_id)
    # fig_id += 1
    # plt.imshow(prediction)
    # plt.axis('off')
    # aurora_demo.save_figure(fig, img_height, img_width, name + "_two_stage", save_folder=save_reults_folder)

    # Save prediction figure of the two-stage process with rotation.
    prediction = aurora_demo.run_on_opencv_image(image, angle=-angle)
    fig = plt.figure(fig_id)
    fig_id += 1
    plt.imshow(prediction)
    plt.axis('off')
    # aurora_demo.save_figure(fig, img_height, img_width, name + "_two_stage_rotation", save_folder=save_reults_folder)

    # # Save original image with the arc normal.
    # fig = plt.figure(fig_id)
    # fig_id += 1
    # plt.imshow(image_ori)
    # plt.axis('off')
    # line_v = np.array([[0, 219.5], [439, 219.5]])
    # sin_a = math.sin(-angle*math.pi/180.)
    # cos_a = math.cos(-angle*math.pi/180.)
    # mtx_r = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    # center_r = np.array([219.5, 219.5])
    # line_r = np.dot(line_v-center_r, mtx_r) + center_r
    # plt.plot(line_r[:, 1], line_r[:, 0], color='red')
    # plt.scatter([219.5], [219.5], s=40, c='r')
    # aurora_demo.save_figure(fig, img_height, img_width, name + "_normal", save_folder=save_reults_folder)

    plt.show()

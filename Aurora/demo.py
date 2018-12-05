import sys
from maskrcnn_benchmark.config_aurora import cfg
from predictor import AuroraDemo
import cv2
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import math


# annFile = '/home/ljm/NiuChuang/AuroraObjectData/annotations_rect_spread_test_spread_416_d60s.json'
# aurora_coco = COCO(annFile)
# imgIds = aurora_coco.getImgIds()

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

show_prediction = True
show_one_stage = True

fig_id = 1
name = 'N20040116G050623'

img_path = './Aurora/N20040116G050623.jpg'

image_ori = cv2.imread(img_path)
img_height = image_ori.shape[0]
img_width = image_ori.shape[1]

save_reults_folder = './demo_results/'
if not os.path.exists(save_reults_folder):
    os.mkdir(save_reults_folder)

# Save figure of original image.
fig = plt.figure(fig_id)
fig_id += 1
plt.imshow(image_ori, cmap='gray')
plt.axis('off')
aurora_demo.save_figure(fig, img_height, img_width, name, save_folder=save_reults_folder)

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

# Save prediction figure of the one-stage process.
prediction = aurora_demo.run_on_opencv_image(image_ori, angle=0)
fig = plt.figure(fig_id)
fig_id += 1
plt.imshow(prediction)
plt.axis('off')
aurora_demo.save_figure(fig, img_height, img_width, name+"_one_stage", save_folder=save_reults_folder)

angle = aurora_demo.compute_angle(image_ori, thresh_bdry_number=150)

if angle is not None:
    image_ori = Image.fromarray(image_ori)

    image = image_ori.rotate(angle)

    image = np.asarray(image)

    angle1 = aurora_demo.compute_angle(image, thresh_bdry_number=150)

if (angle is not None) and (angle1 is not None):

    angle = angle + angle1
    image_r = image_ori.rotate(angle)

    image = np.asarray(image_r)

    # Save the intensity-zangle curve.
    fig = plt.figure(fig_id)
    fig_id += 1
    zenith_angles_i, arc_widths_i = aurora_demo.compute_arc_zangle_width_intensity_line(image, show_details=True, one_fig=True)
    plt.savefig(save_reults_folder+name+'_intensity.png', dpi=300, bbox_inches='tight', transparent=True)

    # Save prediction figure of the two-stage process without rotation.
    prediction = aurora_demo.run_on_opencv_image(image, angle=0)
    fig = plt.figure(fig_id)
    fig_id += 1
    plt.imshow(prediction)
    plt.axis('off')
    aurora_demo.save_figure(fig, img_height, img_width, name + "_two_stage", save_folder=save_reults_folder)

    # Save prediction figure of the two-stage process with rotation.
    prediction = aurora_demo.run_on_opencv_image(image, angle=-angle)
    fig = plt.figure(fig_id)
    fig_id += 1
    plt.imshow(prediction)
    plt.axis('off')
    aurora_demo.save_figure(fig, img_height, img_width, name + "_two_stage_rotation", save_folder=save_reults_folder)

    # Save original image with the arc normal.
    fig = plt.figure(fig_id)
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
    aurora_demo.save_figure(fig, img_height, img_width, name + "_normal", save_folder=save_reults_folder)

    plt.show()

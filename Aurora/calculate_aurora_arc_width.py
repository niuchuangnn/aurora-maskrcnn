import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
import math
from PIL import Image
import math
from skimage import morphology

def real_distance_from_zenith(theta):
    Re = 6370
    h = 150
    alpha = theta - math.asin(Re*math.sin(theta)/(Re+h))
    d = (Re + h) * alpha
    return d

def zenith_anlge(p, mode='radian'):
    zenith = np.array([219.5, 219.5])
    theta_ang = np.linalg.norm(p - zenith) * 90.0 / 246.0
    theta_rad = theta_ang * np.pi / 180.0
    if mode == 'radian':
        return theta_rad
    if mode == 'angle':
        return theta_ang

def distance_aurora(p1, p2):

    theta1 = zenith_anlge(p1)
    theta2 = zenith_anlge(p2)

    d1 = real_distance_from_zenith(theta1)
    d2 = real_distance_from_zenith(theta2)

    d = np.abs(d2-d1)

    return d

def cal_end_points(mask, x):
    v = mask[:, x]
    v_idx = np.where(v == 1)[0]

    if len(v_idx) <= 0:
        return None, None

    v_idx_l = v_idx[0:-1]
    v_idx_r = v_idx[1::]
    v_diff = v_idx_r - v_idx_l
    v_diff_idx = np.where(v_diff > 1)

    if len(v_diff_idx[0]) <= 0:
        v_p1 = np.array([v_idx.min(), 219]).astype(np.float)
        v_p2 = np.array([v_idx.max(), 219]).astype(np.float)
    else:
        if len(v_diff_idx[0]) > 1:
            print(v_diff_idx[0])
        v_min = v_idx.min()
        v_max = v_idx.max()
        v_insert_min = v_idx_l[v_diff_idx[0][0]]
        v_insert_max = v_idx_r[v_diff_idx[0][0]]

        if v_insert_min - v_min >= v_max - v_insert_max:
            v_p1 = np.array([v_min, 219]).astype(np.float)
            v_p2 = np.array([v_insert_min, 219]).astype(np.float)
        else:
            v_p1 = np.array([v_insert_max, 219]).astype(np.float)
            v_p2 = np.array([v_max, 219]).astype(np.float)
    return v_p1, v_p2


mask_folders = '/home/ljm/NiuChuang/AuroraObjectData/mask/'
file_names = '/home/ljm/NiuChuang/AuroraObjectData/images.txt'
mask_type = '.png'

f = open(file_names, 'r')
lines = f.readlines()

color_map = np.random.randint(0, 256, [256, 3])

show_details = False

fig_id = 1
zenith_angles = []
arc_widths = []
for i in range(len(lines)):
    print(i)
    line = lines[i]
    name = line[0:-1]
    img_path = mask_folders + name + mask_type

    mask = imread(img_path)

    mask_color = color_map[mask]

    if show_details:
        plt.figure(fig_id)
        fig_id += 1
        plt.imshow(mask_color)
        plt.axis('off')
        # plt.show()

    labels = list(np.unique(mask))
    labels.pop(labels.index(0))

    for l in labels:
        # mask_l = np.array(mask == l)
        # mask_l = np.uint8(mask_l)
        # contours, hierarchy = cv2.findContours(mask_l, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mask_l = mask == l
        mask_l = morphology.remove_small_objects(mask_l, 20)
        mask_l = np.uint8(mask_l)

        if show_details:
            plt.figure(fig_id)
            fig_id += 1
            plt.imshow(mask_l, cmap='gray')
            plt.axis('off')
            # plt.show()

        u1, contours, u2 = cv2.findContours(mask_l, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = np.squeeze(contours[0])

        if len(contours.shape) < 2:
            num_points = 0
            max_id = 0
            for c in range(len(contours)):
                print(contours[c])
                if contours[c].shape[0] > num_points:
                    num_points = contours[c].shape[0]
                    max_id = c
            contours = np.squeeze(contours[max_id])

        pca = PCA()
        pca.fit(contours)

        components = pca.components_
        main_ax = components[0]
        angle = math.atan(main_ax[1] / main_ax[0]) * (180.0 / math.pi)

        mask_pil = Image.fromarray(np.uint8(mask_l*255))
        mask_pil = mask_pil.rotate(angle)

        if show_details:
            plt.figure(fig_id)
            fig_id += 1
            plt.imshow(mask_pil)
            plt.axis('off')

        mask_h = np.asarray(mask_pil) / 255

        y_219_p1, y_219_p2 = cal_end_points(mask_h, 219)
        y_220_p1, y_220_p2 = cal_end_points(mask_h, 220)

        if y_219_p1 is None or y_220_p1 is None:
            continue

        p1 = (y_219_p1 + y_220_p1) / 2
        p2 = (y_219_p2 + y_220_p2) / 2

        distance = distance_aurora(p1, p2)
        theta1 = zenith_anlge(p1, mode='angle')
        theta2 = zenith_anlge(p2, mode='angle')
        # theta = max(theta1, theta2)
        theta = (theta1+theta2)/2
        p = (p1 + p2) / 2
        if p[0] > 219.5:
            theta = -theta

        if show_details:
            plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'g*')
            plt.title('width: ' + str(distance) + ', angle: ' + str(theta))

        if np.abs(theta) <= 90:
            zenith_angles.append(theta)
            arc_widths.append(distance)

    if show_details:
        plt.show()

plt.figure(fig_id)
fig_id += 1
plt.scatter(zenith_angles, arc_widths, s=2)
plt.show()
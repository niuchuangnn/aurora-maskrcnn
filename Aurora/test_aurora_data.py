from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from PIL import Image
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = '/home/ljm/NiuChuang/AuroraObjectData/'
img_folder = dataDir + 'img/'
annFile = '/home/ljm/NiuChuang/AuroraObjectData/annotations_rect_spread_train_spread_642_d60s.json'
img_type = '.jpg'

aurora = COCO(annFile)

cats = aurora.loadCats(aurora.getCatIds())
nms = [cat['name'] for cat in cats]
print('Aurora categories: \n{}\n'.format(' '.join(nms)))

imgIds = aurora.getImgIds()

for i in range(len(imgIds)):
    img_id = imgIds[i]
    img_path = img_folder + img_id + img_type
    img = aurora.loadImgs([img_id])[0]
    I = Image.open(img_path)
    fig = plt.figure(1)
    plt.axis('off')
    plt.imshow(I, cmap='gray')
    # plt.title('original image')
    ax = plt.gca()

    annIds = aurora.getAnnIds(imgIds=[img['id']])
    anns = aurora.loadAnns(annIds)
    aurora.showAnns(anns)

    for b in range(len(anns)):
        ann = anns[b]
        bbox = ann['bbox']


        xmin = bbox[0]
        ymin = bbox[1]
        bbox_weight = bbox[2]
        bbox_height = bbox[3]

        coords = (xmin, ymin), bbox_weight, bbox_height
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='g', linewidth=3))

        ax.text(xmin, ymin, 'Arc',
                bbox={'facecolor': 'g', 'alpha': 0.5})

    height, width = np.asarray(I).shape
    fig.set_size_inches(width / 20.0 / 3.0, height / 20.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig('test.png', dpi=300)

    # Rotate the image with a random angle.
    fig1 = plt.figure(2)
    angle = np.random.uniform(0, 360)
    # angle = 45
    angle_cos = np.cos(angle*np.pi/180)
    angle_sin = np.sin(angle*np.pi/180)
    # rotate_arr = np.array([angle_sin, angle_cos, angle_cos, -angle_sin]).reshape([2, 2])
    rotate_arr = np.array([angle_cos, angle_sin, -angle_sin, angle_cos]).reshape([2, 2])
    rotate_center = np.array([219.5, 219.5])
    I_r = I.rotate(-angle)
    plt.imshow(I_r)
    plt.axis('off')
    # plt.title('Rotated image, angle: ' + str(angle))
    ax1 = plt.gca()

    # Rotate the annotations.
    for a in range(len(anns)):
        ann = anns[a]
        # Assume the mask has a single component.
        ann_seg = ann['segmentation'][0]

        ann_seg_reshape = np.reshape(np.array(ann_seg), [int(len(ann_seg)/2), 2])
        ann_seg_reshape_rotate = np.dot(ann_seg_reshape-rotate_center, rotate_arr) + rotate_center

        x_min = ann_seg_reshape_rotate[:, 0].min()
        x_max = ann_seg_reshape_rotate[:, 0].max()
        y_min = ann_seg_reshape_rotate[:, 1].min()
        y_max = ann_seg_reshape_rotate[:, 1].max()

        box_rotate = [x_min, y_min, x_max-x_min, y_max-y_min]
        ann_seg_rotate = list(ann_seg_reshape_rotate.flatten())

        anns[a]['bbox'] = box_rotate
        anns[a]['segmentation'][0] = ann_seg_rotate

        coords_r = (x_min, y_min), x_max-x_min, y_max-y_min
        ax1.add_patch(plt.Rectangle(*coords_r, fill=False, edgecolor='g', linewidth=3))

        ax1.text(x_min, y_min, 'Arc',
                bbox={'facecolor': 'g', 'alpha': 0.5})

    aurora.showAnns(anns)

    fig1.set_size_inches(width / 20.0 / 3.0, height / 20.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig('test_r.png', dpi=300)
    # print('sin_angle', angle_sin)

    plt.show()
    pass
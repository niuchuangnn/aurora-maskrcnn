# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
from torchvision import transforms as T
from sklearn.decomposition import PCA
import math
import numpy as np
import numpy
from skimage import morphology
import matplotlib.pyplot as plt
from PIL import Image

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint_aurora import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.boxlist_ops import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

from Aurora.detect_peaks import detect_peaks

from pylab import *


class AuroraDemo(object):
    # COCO categories for pretty print
    CATEGORIES = [
        "__background",
        "Arc",
    ]

    def __init__(
        self,
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        checkpointer = DetectronCheckpointer(cfg, self.model)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image, thresh_mask_area=1000, angle=0):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        top_predictions = self.remove_small_objects(top_predictions, thresh_mask_area=thresh_mask_area)
        top_predictions = self.rotate_prediction(top_predictions, angle=angle)

        image_pil = Image.fromarray(image)
        image_pil = image_pil.rotate(angle)
        image_pil = np.asarray(image_pil)

        result = image_pil.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)

        return result

    def compute_angle(self, image, thresh_bdry_number=150, max=True):
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)
        masks = top_predictions.get_field("mask").numpy()
        if len(masks) <= 0:
            return None
        angles = []
        max_contour = []
        max_contour_num = -1
        for mask in masks:
            thresh = mask[0, :, :, None]
            _, contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                contour_num = contour.shape[0]
                if contour_num > max_contour_num:
                    max_contour = np.squeeze(contour)
                    max_contour_num = contour_num

                if contour_num > thresh_bdry_number:
                    contour = np.squeeze(contour)
                    pca = PCA()
                    pca.fit(contour)

                    components = pca.components_
                    main_ax = components[0]
                    angle = math.atan(main_ax[1] / main_ax[0]) * (180.0 / math.pi)
                    angles.append(angle)

        if len(angles) <= 0 or max:
            pca = PCA()
            pca.fit(max_contour)

            components = pca.components_
            main_ax = components[0]
            angle_ave = math.atan(main_ax[1] / main_ax[0]) * (180.0 / math.pi)
        else:
            angle_ave = np.array(angles).mean()
        return angle_ave

    def compute_arc_masks(self, image, thresh_mask_area=1000):
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        self.remove_small_objects(top_predictions, thresh_mask_area=thresh_mask_area)
        masks = top_predictions.get_field("mask").numpy()

        return masks

    def real_distance_from_zenith(self, theta):
        Re = 6370
        h = 150
        alpha = theta - math.asin(Re * math.sin(theta) / (Re + h))
        d = (Re + h) * alpha
        return d

    def zenith_anlge(self, p, mode='radian'):
        zenith = np.array([219.5, 219.5])
        theta_ang = np.linalg.norm(p - zenith) * 90.0 / 256.0
        theta_rad = theta_ang * np.pi / 180.0
        if mode == 'radian':
            return theta_rad
        if mode == 'angle':
            return theta_ang

    def distance_aurora(self, p1, p2):

        theta1 = self.zenith_anlge(p1)
        theta2 = self.zenith_anlge(p2)

        d1 = self.real_distance_from_zenith(theta1)
        d2 = self.real_distance_from_zenith(theta2)

        if (p1[0]-219.5)*(p2[0]-219.5) < 0:
            d = np.abs(d2+d1)
        else:
            d = np.abs(d2-d1)

        return d

    def cal_end_points(self, mask, x):
        v = mask[:, x]
        v_idx = np.where(v == 1)[0]

        if len(v_idx) <= 0:
            return None, None

        v_idx_l = v_idx[0:-1]
        v_idx_r = v_idx[1::]
        v_diff = v_idx_r - v_idx_l
        v_diff_idx = np.where(v_diff > 1)

        if len(v_diff_idx[0]) <= 0:
            v_p1 = np.array([v_idx.min(), x]).astype(np.float)
            v_p2 = np.array([v_idx.max(), x]).astype(np.float)
        else:
            if len(v_diff_idx[0]) > 1:
                print(v_diff_idx[0])
            v_min = v_idx.min()
            v_max = v_idx.max()
            v_insert_min = v_idx_l[v_diff_idx[0][0]]
            v_insert_max = v_idx_r[v_diff_idx[0][0]]

            if v_insert_min - v_min >= v_max - v_insert_max:
                v_p1 = np.array([v_min, x]).astype(np.float)
                v_p2 = np.array([v_insert_min, x]).astype(np.float)
            else:
                v_p1 = np.array([v_insert_max, x]).astype(np.float)
                v_p2 = np.array([v_max, x]).astype(np.float)
        return v_p1, v_p2

    def smooth(self, x, window_len=11, window='hanning'):
        """smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.

        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        output:
            the smoothed signal

        example:

        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

        see also:

        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter

        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """

        if x.ndim != 1:
            raise(ValueError, "smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise(ValueError, "Input vector needs to be bigger than window size.")

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s = numpy.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = numpy.ones(window_len, 'd')
        else:
            w = eval('numpy.' + window + '(window_len)')

        y = numpy.convolve(w / w.sum(), s, mode='valid')
        return y

    def compute_arc_zangle_width_intensity_line(self, image_rot, show_details=True, one_fig=False):
        if len(image_rot.shape) > 2:
            image_rot = image_rot[:, :, 0]
        image = image_rot.astype(np.float32) / 255.0
        img_width = image.shape[1]
        index_l = int(float(img_width-1) / 2.)
        index_r = int(np.ceil(float(img_width-1) / 2.))
        intensity = (image[:, index_l] + image[:, index_r]) / 2.

        line_coords = [np.array([float(i), float(img_width-1)/2.]) for i in range(img_width)]

        img_height = float(image.shape[0])

        line_zangles = [self.zenith_anlge(p, mode='angle')*((img_height/2.>p[0])*2-1) for p in line_coords]

        window_len = 7
        window_len_half = int((window_len - 1) / 2)
        intensity = self.smooth(intensity, window_len=window_len)
        intensity = list(np.array(intensity)[window_len_half:-window_len_half])

        idx_peaks = detect_peaks(intensity, valley=False, edge='both', show=False)
        idx_valleies = detect_peaks(intensity, valley=True, edge='both', show=False)

        # Search arcs.
        idx_pv = list(idx_peaks) + list(idx_valleies)
        list.sort(idx_pv, reverse=False)

        val_pv = []
        for v in idx_pv:
            if v in idx_peaks:
                val_pv.append(1)

            if v in idx_valleies:
                val_pv.append(-1)

        assert (len(val_pv) == len(idx_pv))

        # Remove the small peaks, namely, the point of intensity difference with its neighbors less than a threshold should be removed.
        idx_keep = []
        intensity_pv = [intensity[v] for v in idx_pv]
        th_arc_remove_l = 0.05
        for v in range(len(idx_pv)):
            if 0 < v < len(idx_pv)-1:
                diff_max = max(np.abs(intensity_pv[v]-intensity_pv[v-1]), np.abs(intensity_pv[v]-intensity_pv[v+1]))
                if diff_max > th_arc_remove_l:
                    idx_keep.append(v)
            else:
                idx_keep.append(v)

        idx_arcs = []
        val_pv = np.array(val_pv)[idx_keep]
        idx_pv = np.array(idx_pv)[idx_keep]
        arc_modes = [np.array([-1, 1, -1]), np.array([-1, 1, 1, -1])]
        # [-1,1,-1] or [-1, 1, 1, -1] represent a arc.
        num_points = len(val_pv)
        for v in range(num_points):
            pv = val_pv[v]
            if pv == -1:

                if v < num_points - 3:
                    if (val_pv[v:v + 3] == arc_modes[0]).all():
                        idx_arcs.append(idx_pv[v:v + 3])

                if v < num_points - 4:
                    if (val_pv[v:v + 4] == arc_modes[1]).all():
                        idx_arcs.append(idx_pv[v:v + 4])

        # Remove the fake arcs.
        idx_arcs_keep = []
        val_arcs_keep = []
        zangle_arcs_keep = []
        zangle_arcs_half_keep = []
        width_arc_keep = []
        th_arc_remove_h = 0.1
        for i in range(len(idx_arcs)):
            idx_arc = idx_arcs[i]
            val_i = np.array(intensity)[idx_arc]

            diff1 = val_i[-2] - val_i[-1]
            diff2 = val_i[1] - val_i[0]
            # condition = (diff1 > th_arc_remove_h or diff2 > th_arc_remove_h) and (diff1 > th_arc_remove_l and diff2 > th_arc_remove_l)
            condition = (diff1 > th_arc_remove_h and diff2 > th_arc_remove_h)
            if condition:
                idx_arcs_keep.append(idx_arc)
                val_arcs_keep.append(val_i)

                # Compute the zenith angle corresponding to half of intensity high.
                an_i = np.array(line_zangles)[idx_arc]
                idx_arc = list(idx_arc)
                list.sort(idx_arc, reverse=False)
                if len(idx_arc) == 3:
                    intensity_half_l = val_i[0:2].mean()
                    intensity_l = np.array(intensity)[range(idx_arc[0], idx_arc[1])]
                    diff = np.abs(intensity_l - intensity_half_l)
                    min_id = np.argmin(diff) + idx_arc[0]
                    an_l = line_zangles[min_id]

                    intensity_half_r = val_i[1:3].mean()
                    intensity_r = np.array(intensity)[range(idx_arc[1], idx_arc[2])]
                    diff = np.abs(intensity_r - intensity_half_r)
                    min_id = np.argmin(diff) + idx_arc[1]
                    an_r = line_zangles[min_id]

                    zangle_arcs_keep.append(an_i[1])
                    zangle_arcs_half_keep.append([an_l, an_r])
                else:
                    intensity_half_l = val_i[0:2].mean()
                    intensity_l = np.array(intensity)[range(idx_arc[0], idx_arc[1])]
                    diff = np.abs(intensity_l - intensity_half_l)
                    min_id = np.argmin(diff) + idx_arc[0]
                    an_l = line_zangles[min_id]

                    intensity_half_r = val_i[2:4].mean()
                    intensity_r = np.array(intensity)[range(idx_arc[2], idx_arc[3])]
                    diff = np.abs(intensity_r - intensity_half_r)
                    min_id = np.argmin(diff) + idx_arc[2]
                    an_r = line_zangles[min_id]

                    zangle_arcs_keep.append(an_i[1:3].mean())
                    zangle_arcs_half_keep.append([an_l, an_r])

                # Compute arc width.
                an1 = zangle_arcs_half_keep[-1][0] * np.pi / 180.
                an2 = zangle_arcs_half_keep[-1][1] * np.pi / 180.
                d1 = self.real_distance_from_zenith(np.abs(an1))
                d2 = self.real_distance_from_zenith(np.abs(an2))
                if an1 * an2 < 0:
                    width_arc_keep.append(np.abs(d1 + d2))
                else:
                    width_arc_keep.append(np.abs(d1 - d2))

        if show_details:
            if not one_fig:
                fig_id = 10000
                plt.figure(fig_id)
                fig_id += 1
                plt.imshow(image, cmap='gray')
                plt.axis('off')

            an_peak = [line_zangles[i] for i in idx_peaks]
            w_peak = [intensity[i] for i in idx_peaks]
            an_valleies = [line_zangles[i] for i in idx_valleies]
            w_valleies = [intensity[i] for i in idx_valleies]

            if not one_fig:
                plt.figure(fig_id)
                fig_id += 1
                plt.plot(line_zangles, intensity)
                plt.scatter(an_peak, w_peak, s=60, marker='o', c='g')
                plt.scatter(an_valleies, w_valleies, s=40, marker='*', c='r')

            if one_fig:
                plt.plot(line_zangles, intensity)

            for i in range(len(idx_arcs_keep)):
                if one_fig:
                    # show on one figure.
                    an_pv_i = [line_zangles[a] for a in idx_arcs_keep[i]]
                    val_pv_i = val_arcs_keep[i]
                    plt.scatter(an_pv_i, val_pv_i, s=60, marker='o', c='g')
                    an_half_i = zangle_arcs_half_keep[i]
                    plt.plot([an_half_i[0], an_half_i[0]], [0, 1], linestyle="--", color='black')
                    plt.plot([an_half_i[1], an_half_i[1]], [0, 1], linestyle="--", color='black')
                    # plt.plot([zangle_arcs_keep[i], zangle_arcs_keep[i]], [0, 1], c='r')
                    plt.xlabel('Zenith angle')
                    plt.ylabel('Intensity')
                else:
                    # show on separate figures.
                    an_pv_i = [line_zangles[a] for a in idx_arcs_keep[i]]
                    val_pv_i = val_arcs_keep[i]
                    plt.figure(fig_id)
                    fig_id += 1
                    plt.plot(line_zangles, intensity)
                    plt.scatter(an_pv_i, val_pv_i, s=60, marker='o', c='g')

                    an_half_i = zangle_arcs_half_keep[i]
                    plt.plot([an_half_i[0], an_half_i[0]], [0, 1])
                    plt.plot([an_half_i[1], an_half_i[1]], [0, 1])
                    plt.plot([zangle_arcs_keep[i], zangle_arcs_keep[i]], [0, 1], c='r')
                    plt.title('Zangle: ' + str(int(np.round(zangle_arcs_keep[i]))) + ', Arc width: ' + str(int(np.round(width_arc_keep[i]))))
                    plt.xlabel('Zenith angle')
                    plt.ylabel('Intensity')

                # plt.show()
        return zangle_arcs_keep, width_arc_keep

    def compute_arc_zangle_width(self, image, thresh_mask_area=1000, thresh_zangle=90, show_details=False):

        masks = self.compute_arc_masks(image, thresh_mask_area=thresh_mask_area)

        zenith_angles = []
        arc_widths = []

        if len(masks) > 0:
            num_instance = masks.shape[0]

            show_details = show_details
            fig_id = 100

            if show_details:
                plt.figure(fig_id)
                fig_id += 1
                plt.imshow(image)
                plt.axis('off')

            for i in range(num_instance):
                mask = masks[i, 0, :, :]

                mask = np.uint8(mask)

                if show_details:
                    plt.figure(fig_id)
                    fig_id += 1
                    plt.imshow(mask, cmap='gray')
                    plt.axis('off')
                    # plt.show()

                u1, contours, u2 = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

                mask_pil = Image.fromarray(np.uint8(mask * 255))
                mask_pil = mask_pil.rotate(angle)

                if show_details:
                    plt.figure(fig_id)
                    fig_id += 1
                    plt.imshow(mask_pil)
                    plt.axis('off')

                mask_h = np.asarray(mask_pil) / 255

                y_219_p1, y_219_p2 = self.cal_end_points(mask_h, 219)
                y_220_p1, y_220_p2 = self.cal_end_points(mask_h, 220)

                if y_219_p1 is None or y_220_p1 is None:
                    continue

                p1 = (y_219_p1 + y_220_p1) / 2
                p2 = (y_219_p2 + y_220_p2) / 2

                distance = self.distance_aurora(p1, p2)
                theta1 = self.zenith_anlge(p1, mode='angle')
                theta2 = self.zenith_anlge(p2, mode='angle')

                theta = (theta1 + theta2) / 2
                p = (p1 + p2) / 2
                if p[0] > 219.5:
                    theta = -theta

                # if p1[0] > p2[0]:
                #     p = p1
                #     theta = theta1
                # else:
                #     p = p2
                #     theta = theta2
                #
                # if p[0] > 219.5:
                #     theta = -theta

                if show_details:
                    plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'g*')
                    plt.title('width: ' + str(distance) + ', angle: ' + str(theta))

                if np.abs(theta) <= thresh_zangle:
                    zenith_angles.append(theta)
                    arc_widths.append(distance)

            if show_details:
                plt.show()

        return zenith_angles, arc_widths

    def rotate_prediction(self, predictions, angle):
        masks = predictions.get_field("mask").numpy()

        for i in range(masks.shape[0]):
            mask = masks[i, 0, :, :] == 1
            mask = np.uint8(mask)

            mask_pil = Image.fromarray(mask)
            mask_pil = mask_pil.rotate(angle)
            mask = np.asarray(mask_pil)

            predictions.extra_fields['mask'][i, 0, :, :] = torch.from_numpy(mask)

        predictions.extra_fields['angle'] = -angle

        return predictions

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            masks = self.masker(masks, prediction)
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def remove_small_objects(self, predictions, thresh_mask_area=1000):
        masks = predictions.get_field("mask").numpy()

        keep_idx = []
        for i in range(masks.shape[0]):
            mask = masks[i, 0, :, :] == 1

            mask = morphology.remove_small_objects(mask, thresh_mask_area)
            mask = np.uint8(mask)

            if mask.max() > 0:
                predictions.extra_fields['mask'][i, 0, :, :] = torch.from_numpy(mask)
                keep_idx.append(i)

            # plt.figure(111)
            # plt.imshow(mask, cmap='gray')
            # plt.show()

        predictions.bbox = predictions.bbox[keep_idx, :]
        predictions.extra_fields['labels'] = predictions.extra_fields['labels'][keep_idx]
        predictions.extra_fields['mask'] = predictions.extra_fields['mask'][keep_idx, :, :, :]
        predictions.extra_fields['scores'] = predictions.extra_fields['scores'][keep_idx]
        return predictions

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        # colors = labels[:, None] * self.palette
        colors = torch.tensor(range(1, len(labels)+1))
        colors = colors[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            if 'angle' not in predictions.extra_fields.keys():
                top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
                image = cv2.rectangle(
                    image, tuple(top_left), tuple(bottom_right), tuple(color), 1
                )
            else:
                angle = predictions.extra_fields['angle']
                rotate_center = torch.tensor([float(image.shape[0] - 1) / 2, float(image.shape[1] - 1) / 2])
                angle_cos = np.cos(angle * np.pi / 180)
                angle_sin = np.sin(angle * np.pi / 180)

                rotate_arr = torch.tensor([angle_cos, angle_sin, -angle_sin, angle_cos]).reshape([2, 2])
                box = box.numpy()
                [xmin, ymin, xmax, ymax] = box[:]
                points = torch.tensor([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]], dtype=torch.float32)

                points_rotate = torch.mm(points - rotate_center, rotate_arr) + rotate_center
                points_rotate = points_rotate.numpy()
                points_rotate = np.int32(points_rotate.reshape((-1, 1, 2)))
                cv2.polylines(image, [points_rotate], True, color, 2)

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None]
            _, contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = torch.nn.functional.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]
        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            if 'angle' not in predictions.extra_fields.keys():
                x, y = box[:2]

            else:
                angle = predictions.extra_fields['angle']
                rotate_center = torch.tensor([float(image.shape[0] - 1) / 2, float(image.shape[1] - 1) / 2])
                angle_cos = np.cos(angle * np.pi / 180)
                angle_sin = np.sin(angle * np.pi / 180)

                rotate_arr = torch.tensor([angle_cos, angle_sin, -angle_sin, angle_cos]).reshape([2, 2])
                xy = torch.mm(box[:2].reshape(1, 2) - rotate_center, rotate_arr) + rotate_center
                x = xy[0, 0]
                y = xy[0, 1]

            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

        return image

    def save_figure(self, fig, img_height, img_width, save_name, save_folder='./results/'):
        fig.set_size_inches(img_width / 30.0 / 3.0, img_height / 30.0 / 3.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(save_folder+save_name+'.png', dpi=300)

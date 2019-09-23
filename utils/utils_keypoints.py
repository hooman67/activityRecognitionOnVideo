import statistics

import cv2
import torch
import torch.backends.cudnn as cudnn
import tensorflow as tf
from tensorflow import Session
import matplotlib.pyplot as plt

from utils.load_pb import load_graph
from utils.utils import convert_bbox_coords_to_pixels
from keypoints.lib.core.config import config, update_config
import keypoints.lib.models as models  # this is called in `eval()`


def init_pose_estimator(cfg, model_file):
    """ Initialize pytorch pose estimator from a config and .tar file"""
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    update_config(cfg)
    model = eval('models.' + config.MODEL.NAME + '.get_pose_net')(
        config, is_train=False)
    if model_file:
        model.load_state_dict(torch.load(model_file))
    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    return model, config


def get_patch_margins_intertooth(boxes, image_w=640, image_h=640, x_factor=1.3, y_bottom_factor=3.):
    """
    Selects margins in pixels to select for detections. Uses intertooth distances.
    :param boxes: list of boxes
    :param image_w:
    :param image_h:
    :param x_factor: multiply the x_margin by x_factor
    :param y_bottom_factor: multiply the y_bottom margin by y_bottom_factor
    :return: 3 integers in pixel distances
    """
    boxes_in_pixels = []
    for box in boxes:
        if box.get_label() == 0:
            xmin, xmax, ymin, ymax = convert_bbox_coords_to_pixels(box,
                                                                   image_w=image_w,
                                                                   image_h=image_h)
            boxes_in_pixels.append([xmin, xmax, ymin, ymax])

    tooth_tips_x = [box[0] + (box[1] - box[0]) / 2 for box in boxes_in_pixels]
    tooth_tips_x = sorted(tooth_tips_x)
    inter_tooth_dists_x = [x2 - x1 for x1, x2 in
                           zip(tooth_tips_x[:-1], tooth_tips_x[1:])]
    if inter_tooth_dists_x:
        median_intertooth_dist_x = statistics.median(inter_tooth_dists_x)
    else:  # if no inter_tooth_dists, i.e. only a single tooth
        median_intertooth_dist_x = 40.

    x_margin = int(x_factor * median_intertooth_dist_x)
    y_top_margin = int(median_intertooth_dist_x)
    y_bottom_margin = int(y_bottom_factor * median_intertooth_dist_x)

    return x_margin, y_top_margin, y_bottom_margin


def get_patch_margins_toothline(boxes, num_teeth, image_w=640, image_h=640, x_factor=1.3, y_bottom_factor=3.):
    """
    Selects margins in pixels to select for detections.
    Divides toothline width by the number of teeth and uses this distance.
    :param boxes:
    :param num_teeth:
    :param image_w:
    :param image_h:
    :param x_factor:
    :param y_bottom_factor:
    :return:
    """
    is_toothline = False
    for box in boxes:
        if box.get_label() == 1:
            is_toothline = True
            xmin, xmax, ymin, ymax = convert_bbox_coords_to_pixels(box,
                                                                   image_w=image_w,
                                                                   image_h=image_h)
    if is_toothline:
        mean_intertooth_dist_x = (xmax - xmin) / num_teeth
        pass
    else:
        mean_intertooth_dist_x = 40.

    x_margin = int(x_factor * mean_intertooth_dist_x)
    y_top_margin = int(mean_intertooth_dist_x)
    y_bottom_margin = int(y_bottom_factor * mean_intertooth_dist_x)

    return x_margin, y_top_margin, y_bottom_margin


def get_tooth_wm_patches(image, boxes, patch_h, patch_w, num_teeth, patch_selection_method="toothline"):
    """ Get patches of objects where each patch should include a tooth and
    corresponding WM landmarks below it.
    Returns:
        patch_coords:list are coords of the patch in the full image
        coord_scales:list in order to map the keypoint coordinates back after resizing
        tooth_tips_from_det:list is center of tooth detection, used for post-processing
    """
    image_h, image_w, _ = image.shape
    patches = []
    patch_coords = []
    coord_scales = []
    tooth_tips_from_det = []
    if patch_selection_method == "intertooth":
        x_margin, y_top_margin, y_bottom_margin = get_patch_margins_intertooth(
            boxes, image_w=image_w, image_h=image_h)
    elif patch_selection_method == "toothline":
        x_margin, y_top_margin, y_bottom_margin = get_patch_margins_toothline(
            boxes, num_teeth, image_w=image_w, image_h=image_h)
    else:
        raise Exception("wrong patch selection method for keypoints")

    for box in boxes:
        if box.get_label() == 0:
            xmin, xmax, ymin, ymax = convert_bbox_coords_to_pixels(box, image_w=image_w,
                                                                   image_h=image_h)
            x_center = xmin + (xmax - xmin) / 2
            y_center = ymin + (ymax - ymin) / 2
            xmin = x_center - x_margin
            ymin = y_center - y_top_margin
            xmax = x_center + x_margin
            ymax = y_center + y_bottom_margin
            xmin = max([0, xmin])
            ymin = max([0, ymin])
            patch = image[int(ymin):int(ymax), int(xmin):int(xmax)]
            x_scale = patch.shape[1] / patch_w
            y_scale = patch.shape[0] / patch_h
            patch = cv2.resize(patch, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)
            patches.append(patch)
            patch_coords.append((xmin, xmax, ymin, ymax))
            coord_scales.append((x_scale, y_scale))

            tooth_tip_from_det = [(xmax - xmin) / (2 * x_scale),
                                  (ymax - ymin) / (2 * y_scale)]
            tooth_tips_from_det.append(tooth_tip_from_det)

    return patches, patch_coords, coord_scales, tooth_tips_from_det


def visualize_keypoints(patches, keypoints):
    """ Plot all patches for debugging"""
    colors = [[128, 0, 128], [0, 255, 0], [0, 0, 255], [255, 0, 0], [153, 255, 255],
              [0, 0, 0]]
    for patch, patch_keypoints in zip(patches, keypoints):
        for i, keypoint in enumerate(patch_keypoints):
            x, y = keypoint
            x, y = int(x), int(y)
            cv2.circle(patch, (x, y), 3, colors[i], -1)
        plt.imshow(patch)
        plt.show()

    return


def map_keypoints(keypoints, patch_coords, coord_scales):
    """ Map predicted keypoints from patch coords to the original image coordinates"""
    image_mapped_keypoints = []
    for patch_keypoints, patch_coord, scale in zip(keypoints, patch_coords, coord_scales):
        patch_mapped_keypoints = []
        if patch_keypoints != []:
            for i, joint_keypoint in enumerate(patch_keypoints):
                x, y = joint_keypoint
                # if x is None or y is None: x, y = 0, 0
                if x == 0 and y == 0:
                    x, y = 0, 0
                else:
                    x *= scale[0]
                    y *= scale[1]
                    x += patch_coord[0]
                    y += patch_coord[2]
                    x, y = int(x), int(y)
                patch_mapped_keypoints.append([x, y])
        image_mapped_keypoints.append(patch_mapped_keypoints)

    return image_mapped_keypoints


def draw_keypoints(image, keypoints, circle_radius=3):
    """ Draw keypoints on the full image."""
    colors = [[128, 0, 128], [0, 255, 0], [0, 0, 255], [255, 0, 0], [153, 255, 255],
              [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for patch_keypoints in keypoints:
        if patch_keypoints == []: continue
        for i, joint_keypoint in enumerate(patch_keypoints):
            x, y = joint_keypoint
            if x == 0 or y == 0:
                continue
            cv2.circle(image, (x, y), circle_radius, colors[i], -1)

    return image


def get_tf_keypoints_network_session_info(tf_model_path):
    """ Returns session (which is coupled with a Graph) and x (which is feed_dict) and y
    (which is fetches)."""
    tf_model = load_graph(tf_model_path)
    x = tf_model.get_tensor_by_name('prefix/input_image:0')
    y = tf_model.get_tensor_by_name('prefix/Add_8:0')
    gpu_options = tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=0.3)
    sess_config = tf.ConfigProto(gpu_options=gpu_options,
                                 use_per_session_threads=False)
    tf_session = Session(graph=tf_model, config=sess_config)

    return tf_session, x, y

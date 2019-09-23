# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import cv2
import numpy as np
import torch
import torchvision.transforms as transf

from utilities.transforms import transform_preds


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array([hm[py][px+1] - hm[py][px-1],
                                     hm[py+1][px]-hm[py-1][px]])
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals


def predict_on_image(image, model, threshold=0.3):
    """ Uses pytorch's Pose-Resnet to predict a map and then decodes it.
    Uses a `threshold` on the values of the map."""
    model.eval()
    _, h, w = image.shape
    image = image[None, ...]  # batch_size of 1
    with torch.no_grad():
        output = model(image)
        preds, maxvals = get_max_preds(output.cpu().numpy())
        preds, maxvals = preds[0], maxvals[0]  # because batch_size of 1
        # Times by 4, because output map resolution is decreased by
        # a factor of with respect to input image
        preds = preds * 4

    for i, maxval in enumerate(maxvals):
        if maxval < threshold:
            preds[i] = [0, 0]

    return preds.tolist()


def predict_on_images(images, model):
    all_keypoints = []
    for image in images:
        image = transf.ToTensor()(image).cuda()
        single_image_keypoints = predict_on_image(image, model)
        all_keypoints.append(single_image_keypoints)

    return all_keypoints


def tf_predict_on_image(image, sess, x, y, image_w, image_h, threshold=0.3):
    image = cv2.resize(image, (image_w, image_h))
    image = image / 255.
    image = image.transpose((2, 0, 1))
    image = image[None, ...]
    output = sess.run(y, {x: image})

    preds, maxvals = get_max_preds(output)
    preds, maxvals = preds[0], maxvals[0]  # because batch_size of 1
    # Times by 4, because output map resolution is decreased by
    # a factor of 4 with respect to input image
    preds = preds * 4

    for i, maxval in enumerate(maxvals):
        if maxval < threshold:
            preds[i] = [0, 0]

    return preds.tolist()


def tf_predict_on_images(images, sess, x, y, image_w, image_h):
    all_keypoints = []
    for image in images:
        keypoints = tf_predict_on_image(image, sess, x, y, image_w, image_h)
        all_keypoints.append(keypoints)

    return all_keypoints

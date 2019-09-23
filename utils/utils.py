import os
import heapq
from time import time

import numpy as np
import cv2

from utils.timestamp import process_xml_to_dict


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None, label=-1):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = label
        self.score = -1

        self.filtered = False

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def bbox_iou_right_join(box1, box2):
    """ Returns right-join ratio IOU (not original IOU)
    Usually box1 would be bucket and box2 can be either Wear Area or MatInside
    """
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    left_join = w2 * h2
    left_join_ratio = float(intersect) / left_join
    return left_join_ratio


def draw_boxes(image, boxes, labels, score_threshold=0.50,
               class_obj_threshold=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3]):
    image_h, image_w, _ = image.shape
    colors = [(0, 255, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
              (0, 255, 255)]

    for box in boxes:
        # if box.get_score() < class_obj_threshold[box.get_label()]: continue
        xmin = int(box.xmin * image_w)
        ymin = int(box.ymin * image_h)
        xmax = int(box.xmax * image_w)
        ymax = int(box.ymax * image_h)
        thickness = 1 if box.filtered else 2
        if box.filtered: continue
        thickness = 2

        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        color = colors[box.get_label()]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color,
                      thickness)
        font_increase = 1.
        cv2.putText(image,
                    # labels[box.get_label()] + ' ' + str(box.get_score())[:4],
                    str(box.get_score())[:4],
                    (xmin, ymin - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    8e-4 * image_h * font_increase,
                    color, thickness)

    return image


def decode_netout(netout, anchors, nb_class, obj_threshold=0.3, nms_threshold=0.01,
                  class_obj_threshold=None):
    if class_obj_threshold is None:
        class_obj_threshold = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    grid_h, grid_w, nb_box = netout.shape[:3]
    boxes = []

    # decode the output by the network
    netout[..., 4] = _sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row, col, b, 5:]

                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row, col, b, :4]

                    x = (col + _sigmoid(x)) / grid_w  # center position, unit: image width
                    y = (row + _sigmoid(
                        y)) / grid_h  # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w  # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h  # unit: image height
                    confidence = netout[row, col, b, 4]

                    box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, confidence,
                                   classes)

                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if
             box.get_score() > class_obj_threshold[box.get_label()]]

    return boxes


def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(
        np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(
        np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]),
                        axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    if np.min(x) < t:
        x = x / np.min(x) * t
    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)


def convert_bbox_coords_to_pixels(bbox, image_w=640, image_h=640):
    xmin = bbox.xmin * image_w
    xmax = bbox.xmax * image_w
    ymin = bbox.ymin * image_h
    ymax = bbox.ymax * image_h
    return xmin, xmax, ymin, ymax


### These are for evaluation of frame selection #########
def TP(ranges, selected_inds):
    tp = 0
    for frame_ind in selected_inds:
        for start, end in ranges:
            if frame_ind >= start and frame_ind <= end:
                tp += 1

    return tp


def FP(ranges, selected_inds):
    fp = 0
    for frame_ind in selected_inds:
        is_false_positive = True
        for start, end in ranges:
            if frame_ind >= start and frame_ind <= end:
                is_false_positive = False
                break
        if is_false_positive:
            fp += 1

    return fp


def FN(ranges, selected_inds):
    fn = 0
    for start, end in ranges:
        is_false_negative = True
        for frame_ind in selected_inds:
            if frame_ind >= start and frame_ind <= end:
                is_false_negative = False
        if is_false_negative:
            fn += 1
    return fn


def precision(ranges, selected_inds):
    """ TP / (TP + FP) """
    tp = TP(ranges, selected_inds)
    fp = FP(ranges, selected_inds)
    if tp == 0:
        return 0
    else:
        return float(tp) / (tp + fp)


def recall(ranges, selected_inds):
    """ TP / (TP + FN) """
    tp = TP(ranges, selected_inds)
    fn = FN(ranges, selected_inds)
    if tp == 0: return 0
    return float(tp) / (tp + fn)


# unit tests
def test_precision():
    ranges = []
    selected_inds = []
    assert precision(ranges, selected_inds) == 0.
    ranges = []
    selected_inds = [1, 100]
    assert precision(ranges, selected_inds) == 0.
    ranges = [(0, 5)]
    selected_inds = [1, 100]
    assert precision(ranges, selected_inds) == 0.5


def test_recall():
    ranges = []
    selected_inds = []
    assert recall(ranges, selected_inds) == 0.
    ranges = [(0, 10)]
    selected_inds = []
    assert recall(ranges, selected_inds) == 0.
    ranges = [(0, 10)]
    selected_inds = [6]
    assert recall(ranges, selected_inds) == 1.
    ranges = [(0, 10), (17, 20), (45, 50)]
    selected_inds = [6, 20, 21]
    assert recall(ranges, selected_inds) == 2. / 3


def evaluate_frame_selection(video_path, pathToTimeStampLabelsXml, selected_inds, log):
    xml_filename = video_path.split("/")[-1]
    xml_filename = xml_filename[:-3] + "xml"
    xml_filepath = os.path.join(pathToTimeStampLabelsXml, xml_filename)

    if os.path.exists(xml_filepath):
        print("EVALUATING ON XML!")
        data = process_xml_to_dict(xml_filepath)

        FM_appropriate = data['FM']
        prec = precision(FM_appropriate, selected_inds)
        rec = recall(FM_appropriate, selected_inds)

        print("\nRecall: %.2f\nPrecision: %.2f" % (prec, rec), file=log)

    return


# For FM frame selection buffers
class MaxHeap(object):
    def __init__(self): self.h = []

    def __len__(self): return len(self.h)

    def heappush(self, x): heapq.heappush(self.h, x)

    def heappop(self): return heapq.heappop(self.h)

    def __getitem__(self, i): return self.h[i]


def tf_predict(sess, y, x, image, image_path, pb_filepath):
    obj_threshold = 0.3
    nms_threshold = 0.01
    image_h, image_w, _ = image.shape
    image = cv2.resize(image, (640, 640))
    image = image / 255.
    # image = self.feature_extractor.normalize(image)

    input_image = image[:, :, ::-1]
    input_image = np.expand_dims(input_image, 0)
    input_image = np.expand_dims(input_image, 0)

    # netout = self.model.predict(input_image)[0]
    start_time = time()
    netout = sess.run(y, {x: input_image})[0]

    anchors = [0.31, 0.81, 0.40, 1.16, 0.41, 0.91, 0.42, 0.68, 0.50, 1.42, 0.52, 1.09,
               0.63, 1.72, 0.96, 1.72, 1.14, 2.69, 10.64, 2.53]
    labels = ["tooth", "toothline"]
    nb_class = len(labels)
    boxes = decode_netout(netout, anchors, nb_class, obj_threshold, nms_threshold)
    print("Time spent: ", time() - start_time)

    image = draw_boxes(image, boxes, labels, obj_threshold)
    # plt.imshow(image)
    # plt.show()
    path_to_save = pb_filepath.split('/')[:-1]
    path_to_save = '/'.join(path_to_save)
    path_to_save = os.path.join(path_to_save, image_path.split('/')[-1])
    print(path_to_save, image.shape)
    image *= 255
    cv2.imwrite(path_to_save, image)

    return boxes

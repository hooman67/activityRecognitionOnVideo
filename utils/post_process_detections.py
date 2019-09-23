import sys
import numpy as np

sys.path.append('../')
from utils.utils import bbox_iou_right_join, convert_bbox_coords_to_pixels

thresholds = \
    {
        "Hydraulic":
            {
                "num_pixel_tooth_threshold": 11,
                "min_area_threshold":
                    {
                        "Tooth": 13 * 13,
                        "Toothline": 200 * 40,
                        "BucketBB": 120 * 120,
                        "MatInside": 200 * 100,
                        "WearArea": 150 * 100,
                        "LipShroud": 18 * 18
                    },
                "max_area_threshold":
                    {
                        "Tooth": 75 * 75,
                        "Toothline": 500 * 250,
                        "BucketBB": 640 * 640,
                        "MatInside": 500 * 500,
                        "WearArea": 400 * 200,
                        "LipShroud": 70 * 50
                    }
            },
        "Cable": {
            "num_pixel_tooth_threshold": 6,
            "min_area_threshold":
                {
                    "Tooth": 12 * 12,
                    "Toothline": 200 * 40,
                    "BucketBB": 110 * 110,
                    "MatInside": 200 * 100,
                    "WearArea": 100 * 100,
                    "LipShroud": 18 * 18
                },
            "max_area_threshold":
                {
                    "Tooth": 50 * 50,
                    "Toothline": 500 * 250,
                    "BucketBB": 640 * 640,
                    "MatInside": 500 * 500,
                    "WearArea": 400 * 300,
                    "LipShroud": 70 * 50
                }
        },
        "Backhoe": {
            "num_pixel_tooth_threshold": 8,
            "min_area_threshold":
                {
                    "Tooth": 12 * 12,
                    "Toothline": 200 * 40,
                    "BucketBB": 150 * 120,
                    "MatInside": 200 * 100,
                    "WearArea": 100 * 100,
                    "LipShroud": 18 * 18
                },
            "max_area_threshold":
                {
                    "Tooth": 50 * 50,
                    "Toothline": 500 * 250,
                    "BucketBB": 640 * 640,
                    "MatInside": 500 * 500,
                    "WearArea": 400 * 300,
                    "LipShroud": 70 * 50
                }
        }
    }

# aspect_ratio = width / height
min_aspect_ratios = {
    "Tooth": 0.28,
    "Toothline": 1.25,
    "BucketBB": 0.7,
    "MatInside": 0.5,
    "WearArea": 1.25,
    "LipShroud": 0.35
}
max_aspect_ratios = {
    "Tooth": 1.4,
    "Toothline": 6.0,
    "BucketBB": 5.0,
    "MatInside": 3.0,
    "WearArea": 4.0,
    "LipShroud": 1.3
}

label_to_ind = {
    "Tooth": 0,
    "Toothline": 1,
    "BucketBB": 2,
    "MatInside": 3,
    "WearArea": 4,
    "LipShroud": 5
}
ind_to_label = {
    0: "Tooth",
    1: "Toothline",
    2: "BucketBB",
    3: "MatInside",
    4: "WearArea",
    5: "LipShroud"
}


def is_object_present(bboxes, class_ind):
    is_present = False
    for bbox in bboxes:
        if bbox.get_label() == class_ind:
            is_present = True

    return is_present


def get_best_bbox_of_class(bboxes, class_index):
    """ Looks for a single highest-probability instance of class class_index """
    obj_to_find_score, obj_to_find_bbox = 0., None
    new_bboxes = []
    filtered_bboxes = []
    for bbox in bboxes:
        if bbox.get_label() == class_index and bbox.get_score() > obj_to_find_score:
            obj_to_find_score = bbox.get_score()
            obj_to_find_bbox = bbox
        elif bbox.get_label() == class_index and bbox.get_score() <= obj_to_find_score:
            filtered_bboxes.append(bbox)
        else:
            new_bboxes.append(bbox)
    if obj_to_find_bbox:
        new_bboxes.append(obj_to_find_bbox)

    return new_bboxes, obj_to_find_bbox, filtered_bboxes


def filter_wrong_sizes(bboxes, image_size, min_area_thresholds, max_area_thresholds,
                       min_aspect_ratios, max_aspect_ratios):
    """ Removes bboxes of wrong area or aspect ratio. """
    image_w, image_h = image_size, image_size

    filtered_bboxes = []
    for bbox in bboxes:
        class_ind = bbox.get_label()
        class_string = ind_to_label[class_ind]
        min_area_threshold = min_area_thresholds[class_string]
        max_area_threshold = max_area_thresholds[class_string]
        min_aspect_ratio = min_aspect_ratios[class_string]
        max_aspect_ratio = max_aspect_ratios[class_string]

        xmin, xmax, ymin, ymax = convert_bbox_coords_to_pixels(bbox, image_w, image_h)
        width = xmax - xmin
        height = ymax - ymin
        bbox_area = width * height
        bbox_aspect = float(width) / height
        if bbox_area < min_area_threshold or bbox_area > max_area_threshold or \
                bbox_aspect < min_aspect_ratio or bbox_aspect > max_aspect_ratio:
            bboxes.remove(bbox)
            filtered_bboxes.append(bbox)

    return bboxes, filtered_bboxes


def filter_teeth_outside_toothline(bboxes, teethline_bbox, x_threshold=40.,
                                   y_threshold=20.):
    # refine tooth bboxes with teethline because teethline dets are more accurate
    filtered_bboxes = []
    if teethline_bbox:
        teethline_bbox_xmin, teethline_bbox_xmax, teethline_bbox_ymin, \
        teethline_bbox_ymax = convert_bbox_coords_to_pixels(teethline_bbox)
        bboxes_to_return = []
        for bbox in bboxes:
            xmin, xmax, ymin, ymax = convert_bbox_coords_to_pixels(bbox)
            if bbox.get_label() == 0:
                if xmin < teethline_bbox_xmin - x_threshold or \
                        xmax > teethline_bbox_xmax + x_threshold or \
                        ymin < teethline_bbox_ymin - y_threshold or \
                        ymax > teethline_bbox_ymax + y_threshold:
                    filtered_bboxes.append(bbox)
                else:
                    bboxes_to_return.append(bbox)
            else:
                bboxes_to_return.append(bbox)
    else:
        bboxes_to_return = []
        filtered_bboxes = bboxes
    return bboxes_to_return, filtered_bboxes


def filter_inside_bucket(bucket_bbox, bbox_to_check, right_join_thresh=0.7):
    if bbox_iou_right_join(bucket_bbox, bbox_to_check) < right_join_thresh:
        return None
    else:
        return bbox_to_check


def filter_close_teeth(bboxes, pixel_threshold=4):
    """ Get rids of FPs that are sometimes predicted between the teeth along the
    toothline. """
    # sort bboxes by x-coordinate
    other_bboxes = [bbox for bbox in bboxes if bbox.get_label() != 0]
    teeth_bboxes = [bbox for bbox in bboxes if bbox.get_label() == 0]
    teeth_xmins = [bbox.xmin for bbox in teeth_bboxes]
    xmin_sorted = np.argsort(teeth_xmins)
    teeth_bboxes = np.array(teeth_bboxes)[xmin_sorted]
    teeth_bboxes = teeth_bboxes.tolist()
    filtered_bboxes = []

    # get rid of bboxes which are within pixel_threshold in the x-axis
    i = 0
    if teeth_bboxes:
        teeth_bboxes_to_return = [teeth_bboxes[0]]
    else:
        teeth_bboxes_to_return = []
    if len(teeth_bboxes) > 2:
        while i + 2 < len(teeth_bboxes):  # iterate over the triples
            bbox1 = teeth_bboxes[i]
            bbox2 = teeth_bboxes[i + 1]
            bbox3 = teeth_bboxes[i + 2]
            xmin1, xmax1, ymin1, ymax1 = convert_bbox_coords_to_pixels(bbox1)
            xmin2, xmax2, ymin2, ymax2 = convert_bbox_coords_to_pixels(bbox2)
            xmin3, xmax3, ymin3, ymax3 = convert_bbox_coords_to_pixels(bbox3)
            if xmin2 - xmax1 < pixel_threshold and xmin3 - xmax2 < pixel_threshold:
                filtered_bboxes.append(bbox2)
            else:
                teeth_bboxes_to_return.append(teeth_bboxes[i + 1])
            i += 1

    if len(teeth_bboxes) > 1:
        teeth_bboxes_to_return.append(teeth_bboxes[-1])

    bboxes_to_return = teeth_bboxes_to_return + other_bboxes
    return bboxes_to_return, filtered_bboxes


def filter_enough_teeth(bboxes, num_teeth):
    # only predict teeth when enough teeth bboxes are observed 
    num_teeth_threshold = num_teeth - 2
    num_of_teeth_dets = 0
    for bbox in bboxes:
        if bbox.get_label() == 0:
            num_of_teeth_dets += 1
    if num_of_teeth_dets <= num_teeth_threshold:
        bboxes = [bbox for bbox in bboxes if bbox.get_label() != 0]

    return bboxes


def filter_low_scoring_teeth(bboxes, teeth_in_bucket=9):
    """ Remove lowest scoring teeth if the detected number of teeth is larger than there
    should be. """
    other_bboxes = [bbox for bbox in bboxes if bbox.get_label() != 0]
    teeth_bboxes = [bbox for bbox in bboxes if bbox.get_label() == 0]
    teeth_scores = [bbox.get_score() for bbox in teeth_bboxes]
    scores_sorted = np.argsort(teeth_scores)
    sorted_teeth_bboxes = np.array(teeth_bboxes)[scores_sorted]
    top_teeth_bboxes = sorted_teeth_bboxes[:teeth_in_bucket].tolist()
    filtered_teeth_bboxes = sorted_teeth_bboxes[teeth_in_bucket:].tolist()

    bboxes = top_teeth_bboxes + other_bboxes
    return bboxes, filtered_teeth_bboxes


def filter_all_objects(bboxes, shovel_type, num_teeth, image_size=640):
    """ Excludes many possible cases of False Positives. """
    image_w, image_h = image_size, image_size
    shovel_thresholds = thresholds[shovel_type]

    all_filtered_bboxes = []
    bboxes, size_filtered_bboxes = filter_wrong_sizes(bboxes, image_size,
                                                      shovel_thresholds[
                                                          "min_area_threshold"],
                                                      shovel_thresholds[
                                                          "max_area_threshold"],
                                                      min_aspect_ratios,
                                                      max_aspect_ratios)
    # seperate the teeth and toothline bounding boxes from bucket, matInside, and wearArea boxes
    teeth_toothline_bboxes = [bbox for bbox in bboxes if bbox.get_label() == 0 or \
                              bbox.get_label() == 1]
    other_bboxes = [bbox for bbox in bboxes if bbox.get_label() != 0 and \
                    bbox.get_label() != 1]
    teeth_toothline_bboxes, tooth_filtered_bboxes = filter_teeth_toothline(
        teeth_toothline_bboxes,
        shovel_thresholds, num_teeth)
    other_bboxes, filtered_other_bboxes = filter_bucket_mat_inside_wear_area(other_bboxes)
    all_filtered_bboxes += size_filtered_bboxes + tooth_filtered_bboxes + \
                           filtered_other_bboxes
    for bbox in all_filtered_bboxes:
        if not bbox:
            all_filtered_bboxes.remove(bbox)
        else:
            bbox.filtered = True

    good_bboxes = teeth_toothline_bboxes + other_bboxes
    return good_bboxes, all_filtered_bboxes


def filter_teeth_toothline(bboxes, shovel_thresholds, num_teeth):
    toothline_class_ind = 1
    new_bboxes, teethline_bbox, filtered_toothlines = get_best_bbox_of_class(
        bboxes, toothline_class_ind)
    new_bboxes, filtered_teeth = filter_teeth_outside_toothline(new_bboxes,
                                                                teethline_bbox)
    # new_bboxes = filter_enough_teeth(new_bboxes, num_teeth)
    new_bboxes, filtered_close_teeth = filter_close_teeth(new_bboxes,
                                                          pixel_threshold=
                                                          shovel_thresholds[
                                                              "num_pixel_tooth_threshold"])
    new_bboxes, filtered_redundant_teeth = filter_low_scoring_teeth(new_bboxes,
                                                                    teeth_in_bucket=num_teeth)
    filtered_teeth_toothline = filtered_toothlines + filtered_teeth + \
                               filtered_close_teeth + filtered_redundant_teeth
    return new_bboxes, filtered_teeth_toothline


def filter_bucket_mat_inside_wear_area(bboxes):
    # find the top single bbox of each Bucket/MatInside/WearArea class
    new_bboxes, bucket_bbox, filtered_bucket_bboxes = get_best_bbox_of_class(
        bboxes, 2)
    new_bboxes, matinside_bbox, filtered_matinside_bboxes = get_best_bbox_of_class(
        new_bboxes, 3)
    new_bboxes, weararea_bbox, filtered_weararea_bboxes = get_best_bbox_of_class(
        new_bboxes, 4)

    filtered_bboxes = []
    if bucket_bbox:
        bboxes_to_return = [bucket_bbox]
    else:
        bboxes_to_return = []

    # make sure matinside and wear area are inside bucket bounding box and only 
    # one of matinside or weararea is detected
    if bucket_bbox and weararea_bbox:
        weararea_bbox = filter_inside_bucket(bucket_bbox, weararea_bbox,
                                             right_join_thresh=0.7)
        if weararea_bbox:
            bboxes_to_return += [weararea_bbox]
            # filtered_bboxes += [matinside_bbox]
    if bucket_bbox and matinside_bbox:  # and not weararea_bbox:
        matinside_bbox = filter_inside_bucket(bucket_bbox, matinside_bbox,
                                              right_join_thresh=0.33)
        if matinside_bbox:
            bboxes_to_return += [matinside_bbox]
        else:
            filtered_bboxes += [matinside_bbox]

    filtered_bboxes += filtered_bucket_bboxes + filtered_matinside_bboxes + \
                       filtered_weararea_bboxes
    return bboxes_to_return, filtered_bboxes

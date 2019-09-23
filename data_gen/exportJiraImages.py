import os
import json
import csv
import argparse
import xml.etree.ElementTree
import xml.etree.cElementTree as ET
import pprint as pp
import math

import numpy as np
import cv2

"""
Requires Python 3

NOTE: Might need to modify the N: path if running from Windows

This file imports a .csv file exported from Jira and produces a folder 'image' with
all images compiled and concatenated json file with all labels.
After creating json file, it creates individual xml file for each image in VOC-like format.

python exportJiraImages.py --csv-file ... --output-dir ... --label-type teeth --is-filter-state --is-visible-part --is-filter-visible --is-debug --is-soft-state
"""

# NOTE:Scene Labeling colors (check version!).
# NOTE: This is old labeling guide, if args.modified-labeling, then it changes in main
rockInside = (0, 0, 255)  # blue     "rock_inside"      = "0000FF"
fineInside = (0, 255, 0)  # green    "fine_inside"      = "00FF00"
emptyInside = (255, 255, 0)  # yellow   "empty"            = "FFFF00"
wmInside = (255, 100, 100)  # L-Pink   "wm_landmarks"     = "FF6464"
inapInside = (150, 100, 50)  # L-Brown  "inapp_For_FM"    = "966432"
teeth = (255, 0, 255)  # pink     "teeth"            = "FF00FF"
case = (255, 0, 0)  # red      "case"             = "FF0000"
truck = (255, 255, 200)  # cream    "truck"            = "FFFFC8"
sheave = (255, 128, 0)  # Orage    "Sheave"           = "FF8000"
cable = (0, 0, 0)  # black    "Cable"            = "000000"
inapForFM = (128, 0, 255)  # purple    "rock_outside"    = "8000FF"
fineOutside = (0, 255, 255)  # cyan      "fine_outside"     = "00FFFF"
void = (180, 50, 50)  # Brown    "void anything not in this labels"  = "B43232"
shadow = (120, 120, 120)  # gray     "shadow inside or outside"          = "787878"
dust = (80, 80, 80)  # D-Gray   "dust inside or outside"            = "505050"

# NOTE: New modified labeling guide
inapForWM = (150, 100, 50)  # L-Brown  "WM Landmarks which is not appropriate for WM"
dustInBucket = (80, 80, 80)  # D-Grat   "Dust inside bucket"

toothline_back = (0, 80, 50)  # toothline for backhoe at an angle, "005032"


def create_folder(directory):
    try:
        os.stat(directory)
    except FileNotFoundError:
        os.makedirs(directory)


# resizes all images to the same size
def apply_crop_setting(image, camera_type):
    original_width = 640
    original_height = 480
    thermal_crop_size_1 = 1280
    thermal_crop_size_2 = 720
    thermal_margin_start_width = 40
    x_shift = 0
    if camera_type == "thermal":
        if image.shape[1] == thermal_crop_size_1:
            image = image[:, :original_width]
        elif image.shape[1] == thermal_crop_size_2:
            image = image[:, thermal_margin_start_width: original_width + \
                                                         thermal_margin_start_width]
            x_shift = thermal_margin_start_width
    x_aspect_ratio = float(original_width) / image.shape[1]
    y_aspect_ratio = float(original_height) / image.shape[0]
    image = cv2.resize(image, (original_width, original_height),
                       interpolation=cv2.INTER_NEAREST)
    return image, x_shift, x_aspect_ratio, y_aspect_ratio


class Initialization:
    def __init__(self, **kwargs):
        self.csv_file = kwargs.get("csv_file")
        self.output_dir = kwargs.get("output_dir")
        self.background_dir = kwargs.get("background_dir")
        self.camera_type = kwargs.get("camera_type")
        self.label_type = kwargs.get("label_type")
        self.is_visible_part = kwargs.get("is_visible_part")
        self.is_filter_state = kwargs.get("is_filter_state")
        self.is_filter_visible = kwargs.get("is_filter_visible")
        self.is_debug = kwargs.get("is_debug")
        self.is_backhoe = kwargs.get("is_backhoe")
        self.is_soft_state = kwargs.get("is_soft_state")

    @property
    def csv_file(self):
        return self._csv_file

    @csv_file.setter
    def csv_file(self, value):
        assert os.path.exists(value), "csv file not found."
        self._csv_file = os.path.normpath(value)

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        create_folder(os.path.normpath(value))
        self._output_dir = os.path.normpath(value)

    @property
    def camera_type(self):
        return self._camera_type

    @camera_type.setter
    def camera_type(self, value):
        assert value.lower() in ["thermal", "optical"], "camera type not valid."
        self._camera_type = value.lower()

    @property
    def label_type(self):
        return self._label_type

    @label_type.setter
    def label_type(self, value):
        assert value.lower() in ["wear", "fragmentation", "teeth", "scene", "teethline",
                                 "sequence_teeth"], "script does not support declared label type "
        self._label_type = value.lower()


class ReadCSVFile:
    def __init__(self, csv_file, label_type):
        self.csv_file = csv_file
        self.column_name = self.get_column_name(label_type)

    @staticmethod
    def get_column_name(label_type):
        if label_type in ["teethline", "teeth"]:
            column_name = "Teeth"
        else:
            column_name = "Scene"
        return column_name

    @property
    def read_csv(self):
        label_folders = []
        video_paths = []
        with open(self.csv_file, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            first_row = next(reader)
            index_label_directory = first_row.index(
                "Custom field ({0} - Image Directory)".format(self.column_name))
            index_is_labeled = first_row.index(
                "Custom field ({0} Images Labeled)".format(self.column_name))
            index_video_path = first_row.index("Custom field (Full Path)")
            for row in reader:
                if row[index_is_labeled] == "Yes":
                    # label_folder = row[index_label_directory].replace('file:////motionmetrics.net/nas/', 'N:/')
                    label_folder = row[index_label_directory].replace(
                        'file:////motionmetrics.net/nas/', '/media/sf_N_DRIVE/')
                    video_path = row[index_video_path].replace(
                        'file:////motionmetrics.net/nas/', '/media/sf_N_DRIVE/')
                    video_paths.append(video_path)
                    label_folders.append(os.path.normpath(label_folder))
        return label_folders, video_paths


class TeethlineTeeth(ReadCSVFile):
    def __init__(self, args):
        self.label_type = args.label_type
        self.camera_type = args.camera_type
        self.output_dir = args.output_dir
        self.is_visible_part = args.is_visible_part
        self.is_filter_state = args.is_filter_state
        self.is_filter_visible = args.is_filter_visible
        self.x_shift = 0
        self.x_aspect_ratio = 1
        self.y_aspect_ratio = 1
        self.is_debug = args.is_debug
        self.is_backhoe = args.is_backhoe
        self.is_soft_state = args.is_soft_state
        ReadCSVFile.__init__(self, args.csv_file, self.label_type)

    @property
    def x_shift(self):
        return self._x_shift

    @x_shift.setter
    def x_shift(self, value):
        self._x_shift = value

    @property
    def x_aspect_ratio(self):
        return self._x_aspect_ratio

    @x_aspect_ratio.setter
    def x_aspect_ratio(self, value):
        self._x_aspect_ratio = value

    @property
    def y_aspect_ratio(self):
        return self._y_aspect_ratio

    @y_aspect_ratio.setter
    def y_aspect_ratio(self, value):
        self._y_aspect_ratio = value

    def assert_coords(self, xmin, ymin, xmax, ymax, angle):
        # if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0 or xmin > xmax or ymin > ymax or xmax > 640+10 or ymax > 480+10: # or angle > 180 or angle < -180:
        #     print("Wrong Coordinates!", xmin, ymin, xmax, ymax, angle)
        #     return False
        # else:
        return True

    def parse_xml(self, xml_file):
        try:
            tree = xml.etree.ElementTree.parse(xml_file)
            root = tree.getroot()
        except Exception as message:
            print("Couldn't open xml_file")
            print(message)
            return None
        return root

    def soft_clip_coords(self, xmin, ymin, xmax, ymax):
        if xmin > -7 and xmin < 0: xmin = 0
        if ymin > -7 and ymin < 0: ymin = 0

        return xmin, ymin, xmax, ymax

    def get_boundaries(self, imgLabel, class_name="BucketBB"):
        '''
        HS: 
        ---This method combines all the labels that are inside the bucket to find the bucket boundaries which 
        exlude Teeth and Case.  
        '''
        if class_name == "BucketBB":
            labelsDic = {
                # 'rock':np.where(np.all(imgLabel == rockInside, axis=-1)),
                'empty': np.where(np.all(imgLabel == emptyInside, axis=-1)),
                'wm': np.where(np.all(imgLabel == wmInside, axis=-1)),
                # 'shad':np.where(np.all(imgLabel == shadow, axis=-1)),
                'case': np.where(np.all(imgLabel == case, axis=-1)),
                'sheave': np.where(np.all(imgLabel == sheave, axis=-1)),
                'toothline_back': np.where(np.all(imgLabel == toothline_back, axis=-1)),
                # 'dust':np.where(np.all(imgLabel == dust, axis=-1)),
                # 'fine':np.where(np.all(imgLabel == fineInside, axis=-1)),
                # 'inap':np.where(np.all(imgLabel == inapInside, axis=-1))
                # 'inapFM':np.where(np.all(imgLabel == inapForFM, axis=-1)),
                # 'dustInBucket':np.where(np.all(imgLabel == dustInBucket, axis=-1)),
                'inapWM': np.where(np.all(imgLabel == inapForWM, axis=-1)),
                'teeth': np.where(np.all(imgLabel == teeth, axis=-1))
            }
            cableDic = {"cable": np.where(np.all(imgLabel == cable, axis=-1))}
        elif class_name == "MatInside":  # material inside for FM
            labelsDic = {
                'rock': np.where(np.all(imgLabel == rockInside, axis=-1)),
                'shad': np.where(np.all(imgLabel == shadow, axis=-1)),
                # 'dust':np.where(np.all(imgLabel == dust, axis=-1)),
                'fine': np.where(np.all(imgLabel == fineInside, axis=-1)),
                # 'inap':np.where(np.all(imgLabel == inapInside, axis=-1))
                'inapFM': np.where(np.all(imgLabel == inapForFM, axis=-1)),
                # 'dustInBucket':np.where(np.all(imgLabel == dustInBucket, axis=-1)),
            }
            if len(labelsDic['shad'][0]) > 125 ** 2:
                labelsDic.pop('shad', None)
            if len(labelsDic['rock'][0]) < 70 ** 2:
                labelsDic.pop('rock', None)

            if sum([len(labelsDic[key][0]) for key in
                    list(labelsDic.keys())]) < 150 * 125:
                labelsDic = {}
        elif class_name == "WearArea":
            labelsDic = {
                'wm': np.where(np.all(imgLabel == wmInside, axis=-1)),
                # 'inapWM':np.where(np.all(imgLabel == inapForWM, axis=-1))
            }
            if len(labelsDic['wm'][0]) < 150 * 50:
                labelsDic = {}
        else:
            raise Exception("SMTH WRONG!")

        # This flags signals whether we could detect any boundaries or not
        foundBucketBoundary = False
        for k in list(labelsDic.keys()):
            if len(labelsDic[k][0]) > 0:
                foundBucketBoundary = True

        # This flag can be used to label the boundary as either full, or empty bucket
        # isFullBucket = len(labelsDic['rock'][0]) > 0

        # get the boundary for each label
        sortedLabelColsDic = {}
        sortedLabelRowsDic = {}
        for k in list(labelsDic.keys()):
            if len(labelsDic[k][0]) > 0:
                sortedLabelColsDic[k + 'Cols'] = np.sort(labelsDic[k][0]),
                sortedLabelRowsDic[k + 'Rows'] = np.sort(labelsDic[k][1]),

        # Get the min and max for rows and columns to caluclate bucket boundaries
        lowColVals = []
        highColVals = []
        for labV in list(sortedLabelColsDic.values()):
            for v in labV:
                lowColVals.append(v[0])
                highColVals.append(v[len(v) - 1])

        lowRowVals = []
        highRowVals = []
        for labV in list(sortedLabelRowsDic.values()):
            for v in labV:
                lowRowVals.append(v[0])
                highRowVals.append(v[len(v) - 1])

        # get the bucket boundary edges
        ymins = np.sort(np.array(lowColVals))
        ymaxs = np.sort(np.array(highColVals))
        xmins = np.sort(np.array(lowRowVals))
        xmaxs = np.sort(np.array(highRowVals))

        if (foundBucketBoundary):
            # return final bucket boundary points
            xmin, xmax, ymin, ymax = xmins[0], xmaxs[len(xmaxs) - 1], ymins[0], ymaxs[
                len(ymaxs) - 1]
            if class_name == "BucketBB":
                sortedLabelColsDic = {}
                sortedLabelRowsDic = {}
                for k in list(cableDic.keys()):
                    if len(cableDic[k][0]) > 0:
                        sortedLabelColsDic[k + 'Cols'] = np.sort(cableDic[k][0]),
                        sortedLabelRowsDic[k + 'Rows'] = np.sort(cableDic[k][1]),

                # Get the min and max for rows and columns to caluclate bucket boundaries
                lowColVals = []
                highColVals = []
                for labV in list(sortedLabelColsDic.values()):
                    for v in labV:
                        lowColVals.append(v[0])
                        highColVals.append(v[len(v) - 1])
                lowRowVals = []
                highRowVals = []
                for labV in list(sortedLabelRowsDic.values()):
                    for v in labV:
                        lowRowVals.append(v[0])
                        highRowVals.append(v[len(v) - 1])
                # get the bucket boundary edges
                ymins_cable = np.sort(np.array(lowColVals))
                ymaxs_cable = np.sort(np.array(highColVals))
                if ymaxs_cable:
                    ymin = int(np.min([ymin, ymaxs_cable[len(ymaxs_cable) - 1]]))

            width, height = xmax - xmin, ymax - ymin
            if width * height < 100 * 50:
                return False, "", "", "", "",
            return foundBucketBoundary, xmin, xmax, ymin, ymax
        else:
            return foundBucketBoundary, "", "", "", "",

    def add_label(self, image_info_dict, exported_image_name, xmin, xmax, ymin, ymax,
                  class_name):
        image_info_dict[exported_image_name][class_name] = {}
        image_info_dict[exported_image_name][class_name]["Y_CanvasTop"] = \
            int(round(self.y_aspect_ratio * ymin))
        image_info_dict[exported_image_name][class_name]["X_CanvasLeft"] = \
            int(round(self.x_aspect_ratio * (xmin - self.x_shift)))
        image_info_dict[exported_image_name][class_name]["Height"] = \
            int(round(self.y_aspect_ratio * (ymax - ymin)))
        image_info_dict[exported_image_name][class_name]["Width"] = \
            int(round(self.x_aspect_ratio * (xmax - xmin)))

        image_info_dict[exported_image_name][class_name]["state"] = 2
        return image_info_dict

    def correct_bbox_w_angle(self, xmin, ymin, xmax, ymax, angle):
        """ Only modifies the x-coordinate in order to include the whole tooth in
        the case of rotation. """
        width, height = xmax - xmin, ymax - ymin
        angle_in_rad = math.radians(angle)
        if angle > 1:
            xmin = xmin - height * math.sin(angle_in_rad)
        elif angle < -1:
            xmax = xmax - height * math.sin(angle_in_rad)
        return int(round(xmin)), ymin, int(round(xmax)), ymax

    def get_object_info_dict(self, info, x_canvas_left, y_canvas_top, height, width):
        """ Extract bbox info, do some pre-processing and put it into a dict. """
        y_object_canvas_top = int(round(y_canvas_top * self.y_aspect_ratio))
        x_object_canvas_left = int(round(x_canvas_left - self.x_shift) *
                                   self.x_aspect_ratio)
        object_height = int(round(height * self.y_aspect_ratio))
        object_width = int(round(width * self.x_aspect_ratio))
        rotate_angle = int(info.find("RotateAngle").text)
        x_object_canvas_left, y_object_canvas_top, xmax, ymax = \
            self.correct_bbox_w_angle(x_object_canvas_left,
                                      y_object_canvas_top,
                                      object_width + x_object_canvas_left,
                                      object_height + y_object_canvas_top, rotate_angle)
        x_object_canvas_left, y_object_canvas_top, xmax, ymax = \
            self.soft_clip_coords(x_object_canvas_left, y_object_canvas_top,
                                  xmax, ymax)
        object_width = xmax - x_object_canvas_left
        object_height = ymax - y_object_canvas_top

        is_wrong = not self.assert_coords(x_object_canvas_left,
                                          y_object_canvas_top,
                                          x_object_canvas_left + object_width,
                                          y_object_canvas_top + object_height,
                                          rotate_angle)

        state = info.find("State")
        if state is None:
            state = -1
        else:
            state = int(info.find("State").text)
        object_info_dict = {"Y_CanvasTop": y_object_canvas_top,
                            "X_CanvasLeft": x_object_canvas_left,
                            "Height": object_height,
                            "Width": object_width,
                            "rotate_angle": rotate_angle,
                            "state": state}

        return object_info_dict, is_wrong

    def filter_state(self, info, excluded_states=[-1, 0]):
        """ Get rid of objects which are excluded_states. """
        is_continue = False
        state = info.find("State")
        if state is None:
            pass
        else:
            state = int(info.find("State").text)
            if state in excluded_states:  # exclude occluded objects
                # print("Excluding tooth bbox because its state is zero!!!")
                is_continue = True

        return is_continue

    def get_only_visible_part(self, info):
        """ Take the height of the object to only be above the yellow line. """
        landmarks = info.findall("Landmarks")
        for landmark in landmarks:
            height = 0
            height_Y = int(float(landmark.find("Y").text))
            if height_Y != 0:
                height = float(landmark.find("Y").text)
        if landmarks == [] or height == 0:
            height = float(info.find("Height").text)
        return height

    def filter_small_visible_part(self, info, width, height, ratio_thresh=0.25,
                                  min_area_threshold=11 * 11, image_name=""):
        """ Get rid of objects whose visible part (above yellow line) is too small.
        """
        is_continue = False

        # Filter the teeth whose visible part is <threshold of the total tooth height
        height_full = float(info.find("Height").text)
        if height / height_full < ratio_thresh:
            print("Small visible area of %s in %s: %.1f / %.1f" % \
                  (info.find("Label").text, image_name, height, height_full))
            is_continue = True

        # Filter the teeth whose area is smaller than threshold
        if width * height < min_area_threshold:
            print("\t======Object too small %s, Area: %.1f" % \
                  (info.find("Label").text, width * height))
            is_continue = True

        return is_continue

    def extract_teethline_teeth_info(self, xml_image_info, exported_image_name,
                                     image_info_dict, is_filter_state, is_visible_part,
                                     is_filter_visible):
        """
        Extracts tooth coords and puts it into dict.
        Coords of toothline are inferred from teeth coords.

        @param is_visible_par: take only visible part of each tooth
        @param is_filter_state: exclude occluded teeth (i.e. state of zero or one)
        @param is_filter_visible: exclude teeth that are partially visible (full bucket)
        
        NOTE: Teethline computation doesn't take into account is_visible_part and
        is_filter state!!! It computes teethline of all teeth which are labeled.
        """
        min_tooth_line_width = 75
        min_toothline_height = 40

        min_y_canvas_top, max_y_canvas_top = 1000, 0
        min_x_canvas_left, max_x_canvas_left = 1000, 0
        height_min_y_canvas_top, height_min_x_canvas_left = 0, 0
        image_info_dict[exported_image_name]["Tooth"] = []
        image_info_dict[exported_image_name]["Toothline"] = {}
        teethInfo = []
        is_wrong_any = False
        for info in xml_image_info.findall('Container'):
            if info.find("Label").text != "Tooth": continue

            y_canvas_top = float(info.find("Y_CanvasTop").text)
            x_canvas_left = float(info.find("X_CanvasLeft").text)
            width = float(info.find("Width").text)
            height = float(info.find("Height").text)
            if self.is_backhoe: y_canvas_top -= height + 2

            if min_y_canvas_top > y_canvas_top:
                min_y_canvas_top = y_canvas_top
            if max_y_canvas_top < y_canvas_top:
                max_y_canvas_top = y_canvas_top
                height_min_y_canvas_top = height
            if min_x_canvas_left > x_canvas_left:
                min_x_canvas_left = x_canvas_left
            if max_x_canvas_left < x_canvas_left:
                max_x_canvas_left = x_canvas_left
                height_min_x_canvas_left = width

            if is_filter_state:
                is_continue = self.filter_state(info, excluded_states=[0, 1])
                if is_continue: continue

            if is_visible_part:
                height = self.get_only_visible_part(info)
            else:
                height = float(info.find("Height").text)

            if is_filter_visible:
                is_continue = self.filter_small_visible_part(info, width, height,
                                                             image_name=exported_image_name)
                if is_continue: continue

            # is_wrong refers to the coordinates of the objects being weird (outside image, etc)
            tooth_info, is_wrong = self.get_object_info_dict(info,
                                                             x_canvas_left, y_canvas_top,
                                                             height, width)
            if is_wrong:
                is_wrong_any = True
                continue

            teethInfo.append(tooth_info)

        if teethInfo == []:
            is_wrong_any == True

        y_toothline_canvas_top = int(round(self.y_aspect_ratio * min_y_canvas_top))
        toothline_height = int(round(self.y_aspect_ratio *
                                     (
                                                 max_y_canvas_top + height_min_y_canvas_top - min_y_canvas_top)))
        x_toothline_canvas_left = int(round(self.x_aspect_ratio *
                                            (min_x_canvas_left - self.x_shift)))
        tooth_line_width = int(round(self.x_aspect_ratio *
                                     (max_x_canvas_left + height_min_x_canvas_left - min_x_canvas_left)))
        image_info_dict[exported_image_name]["Tooth"] = teethInfo
        if tooth_line_width * toothline_height > min_tooth_line_width * min_toothline_height \
                and not is_wrong_any and y_toothline_canvas_top != 1000:
            image_info_dict[exported_image_name]["Toothline"]["Y_CanvasTop"] = \
                y_toothline_canvas_top
            image_info_dict[exported_image_name]["Toothline"]["Height"] = \
                toothline_height
            image_info_dict[exported_image_name]["Toothline"]["X_CanvasLeft"] = \
                x_toothline_canvas_left
            image_info_dict[exported_image_name]["Toothline"]["Width"] = \
                tooth_line_width
            image_info_dict[exported_image_name]["Toothline"]["state"] = 2
        else:
            # print("\t============= Small toothline. Area: ",
            #         tooth_line_width * toothline_height)
            is_wrong_any = True
        return image_info_dict, is_wrong_any

    def exclude_matInside_wrt_bucket(self, image_info_dict, exported_image_name):
        if "BucketBB" in image_info_dict[exported_image_name] and "MatInside" in \
                image_info_dict[exported_image_name]:

            bucket = [image_info_dict[exported_image_name]["BucketBB"]["X_CanvasLeft"],
                      image_info_dict[exported_image_name]["BucketBB"]["Y_CanvasTop"],
                      image_info_dict[exported_image_name]["BucketBB"]["Width"],
                      image_info_dict[exported_image_name]["BucketBB"]["Height"]]

            mat_inside = [
                image_info_dict[exported_image_name]["MatInside"]["X_CanvasLeft"],
                image_info_dict[exported_image_name]["MatInside"]["Y_CanvasTop"],
                image_info_dict[exported_image_name]["MatInside"]["Width"],
                image_info_dict[exported_image_name]["MatInside"]["Height"]]

            area_bucket = bucket[2] * bucket[3]
            area_mat_inside = mat_inside[2] * mat_inside[3]

            if float(area_mat_inside) / area_bucket < 0.33:
                print("BUCKET EXCLUDED!")
                del image_info_dict[exported_image_name]["MatInside"]
                image_info_dict[exported_image_name]["folder_name"] = "debug_problematic"

        return image_info_dict

    def extract_other_objects(self, xml_image_info, exported_image_name, image_info_dict,
                              is_filter_state, is_visible_part, is_filter_visible):
        """ For BucketBB, MatInside, WearArea, LipShroud. """
        other_objects = ["BucketBB", "MatInside", "WearArea"]  # , "LipShroud"]
        image_info_dict[exported_image_name]["LipShroud"] = []
        image_info_dict[exported_image_name]["folder_name"] = "debug"
        min_area_thresholds = {"LipShroud": 16 * 16,
                               "BucketBB": 100 * 100,
                               "MatInside": 100 * 100,
                               "WearArea": 100 * 50}
        lip_shrouds_info = []
        is_wrong_any = False
        for info in xml_image_info.findall('Container'):
            label = info.find("Label").text
            if label not in other_objects: continue

            y_canvas_top = float(info.find("Y_CanvasTop").text)
            x_canvas_left = float(info.find("X_CanvasLeft").text)
            width = float(info.find("Width").text)
            height = float(info.find("Height").text)
            object_info, is_wrong = self.get_object_info_dict(info,
                                                              x_canvas_left,
                                                              y_canvas_top,
                                                              height, width)
            if is_wrong:
                is_wrong_any = True
                continue

            if is_filter_state:
                if label == "MatInside":
                    excluded_states = [0]
                elif label == "WearArea":
                    excluded_states = [0]
                elif label == "BucketBB":
                    excluded_states = []
                else:
                    print('Error unknown label in data. extract_other_objects method')
                is_continue = self.filter_state(info, excluded_states=excluded_states)
                if is_continue: continue

            if is_visible_part:
                height = self.get_only_visible_part(info)
            else:
                height = float(info.find("Height").text)

            if is_filter_visible:
                is_continue = self.filter_small_visible_part(info, width, height,
                                                             min_area_threshold=
                                                             min_area_thresholds[label])
                if is_continue:
                    continue

            if label == "LipShroud":
                lip_shrouds_info.append(object_info)
            elif label == "BucketBB" or label == "MatInside" or label == "WearArea":
                image_info_dict[exported_image_name][label] = object_info

        image_info_dict[exported_image_name]["LipShroud"] = lip_shrouds_info
        image_info_dict = self.exclude_matInside_wrt_bucket(image_info_dict,
                                                            exported_image_name)

        return image_info_dict, is_wrong_any

    def visualize_img(self, image, image_info_dict, exported_image_name, save_path,
                      original_image=None):
        """ Should plot images with bboxes for debugging in a seperate folder. """
        folder_name = ""
        for bbox_name, bboxes in list(image_info_dict[exported_image_name].items()):
            color = (255, 0, 0)
            thickness = 1
            if bbox_name != "Tooth" and bbox_name != "LipShroud": bboxes = [bboxes]
            if bbox_name == "WearArea":
                color = (0, 255, 0)
                thickness = 1
            elif bbox_name == "LipShroud":
                color = (255, 255, 0)
            elif bbox_name == "BucketBB" or bbox_name == "Bucket":
                color = (125, 255, 0)
                thickness = 3
            elif bbox_name == "MatInside" or bbox_name == "WearArea":
                color = (125, 55, 255)
            for bbox in bboxes:
                if type(bbox) is not dict or bbox == {}: continue
                xmin = bbox['X_CanvasLeft']
                ymin = bbox['Y_CanvasTop']
                xmax = xmin + bbox['Width']
                ymax = ymin + bbox['Height']
                state = bbox['state']
                if state == 0 or state == 1:
                    folder_name = "debug_problematic"
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
                cv2.putText(image,
                            str(state),
                            (xmin, ymin - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            8e-4 * 640,
                            color, thickness)
                if original_image is not None:
                    cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), color,
                                  thickness)

        if original_image is not None:
            image = np.concatenate((image, original_image), axis=1)
        if not folder_name:
            folder_name = image_info_dict[exported_image_name]["folder_name"]
        path = os.path.join(save_path, folder_name, exported_image_name)
        cv2.imwrite(path, image)

    def filter_matInside_outside_bucket(self, image_info_dict, exported_image_name):
        """ Makes the MatInside to be within the area of the Bucket. """
        bucket = [image_info_dict[exported_image_name]["BucketBB"]["X_CanvasLeft"],
                  image_info_dict[exported_image_name]["BucketBB"]["Y_CanvasTop"],
                  image_info_dict[exported_image_name]["BucketBB"]["Width"],
                  image_info_dict[exported_image_name]["BucketBB"]["Height"]]

        mat_inside = [
            image_info_dict[exported_image_name]["MatInside"]["X_CanvasLeft"],
            image_info_dict[exported_image_name]["MatInside"]["Y_CanvasTop"],
            image_info_dict[exported_image_name]["MatInside"]["Width"],
            image_info_dict[exported_image_name]["MatInside"]["Height"]]

        if (mat_inside[0] + mat_inside[2] < bucket[0]) or \
                (bucket[0] + bucket[2] < mat_inside[0]) or \
                (mat_inside[1] > bucket[1] + bucket[3]):
            print("\n=====Deleting matinside!")
            del image_info_dict[exported_image_name]["MatInside"]
            return image_info_dict

        xmin = np.max([bucket[0], mat_inside[0]])
        ymin = np.max([bucket[1], mat_inside[1]])
        xmax = np.min([bucket[0] + bucket[2], mat_inside[0] + mat_inside[2]])
        ymax = np.min([bucket[1] + bucket[3], mat_inside[1] + mat_inside[3]])
        width = xmax - xmin
        height = ymax - ymin

        image_info_dict[exported_image_name]["MatInside"]["X_CanvasLeft"] = int(xmin)
        image_info_dict[exported_image_name]["MatInside"]["Y_CanvasTop"] = int(ymin)
        image_info_dict[exported_image_name]["MatInside"]["Width"] = int(width)
        image_info_dict[exported_image_name]["MatInside"]["Height"] = int(height)
        return image_info_dict

    def process_teeth(self, root, image_folder, save_path, image_info_dict, video_path,
                      label_type="teeth"):
        """ Iterates through each image in a folder and extracts bbox info, filter and puts it into a dict. """
        is_wrong = False
        if label_type == "sequence_teeth" and \
                len(root.findall('XMLSaveThumbnail')) % 7 != 0:
            is_wrong = True
            print("WRONG")
            return image_info_dict, is_wrong

        for xml_image in root.findall('XMLSaveThumbnail'):
            original_image_name = xml_image.get("Path")
            exported_image_name = original_image_name.replace(".png", ".jpg")
            image = cv2.imread(os.path.join(image_folder, original_image_name))
            if image is None:
                print(image_folder, original_image_name)
                image = cv2.imread(os.path.join(image_folder,
                                                original_image_name.replace(".png",
                                                                            ".jpg")))
            image, self.x_shift, self.x_aspect_ratio, self.y_aspect_ratio = \
                apply_crop_setting(image, self.camera_type)
            image_info_dict[exported_image_name] = {}
            image_info_dict, is_wrong = \
                self.extract_teethline_teeth_info(xml_image,
                                                  exported_image_name,
                                                  image_info_dict,
                                                  self.is_filter_state,
                                                  self.is_visible_part,
                                                  self.is_filter_visible)
            image_info_dict[exported_image_name]["folder_name"] = "debug"
            image_info_dict, is_wrong = \
                self.extract_other_objects(xml_image, exported_image_name,
                                           image_info_dict,
                                           self.is_filter_state,
                                           self.is_visible_part,
                                           self.is_filter_visible)
            image_info_dict[exported_image_name]["VideoPath"] = video_path
            has_container = xml_image.find("HasContainer").text
            is_wrong = is_wrong or not has_container

            if not is_wrong:
                cv2.imwrite(os.path.join(save_path, exported_image_name), image)

                if self.is_debug:
                    image_debug = image.copy()
                    self.visualize_img(image_debug, image_info_dict,
                                       exported_image_name, save_path)

            if is_wrong:
                image_info_dict.pop(exported_image_name, None)
                image_info_dict[exported_image_name] = {}
                image_info_dict[exported_image_name]["VideoPath"] = video_path

        return image_info_dict, is_wrong

    def process_scene(self, image_folder, save_path, image_info_dict):
        teeth_images = os.listdir(image_folder)

        teeth_images = [img_name for img_name in teeth_images if
                        (img_name.endswith('.png') or img_name.endswith('.jpg'))]

        data_folder = os.path.abspath(os.path.join(image_folder, "../.."))

        scene_label_dir = os.path.join(data_folder, "Scene", "Label", "PNG")

        for scene_image_name in os.listdir(scene_label_dir):

            if not scene_image_name.endswith(".png"):
                continue

            if scene_image_name not in teeth_images:
                print(scene_image_name)
                continue

            exported_image_name = scene_image_name.replace(".png", ".jpg")
            if exported_image_name not in list(image_info_dict.keys()):
                continue

            scene_image = cv2.imread(os.path.join(scene_label_dir, scene_image_name))
            if scene_image is None:
                scene_image = cv2.imread(os.path.join(scene_label_dir,
                                                      scene_image_name))
            scene_image = cv2.cvtColor(scene_image, cv2.COLOR_BGR2RGB)
            # if exported_image_name not in image_info_dict.keys(): 
            #     image_info_dict[exported_image_name] = {}
            for class_name in ["BucketBB", "MatInside", "WearArea"]:
                is_found_bucket, xmin, xmax, ymin, ymax = \
                    self.get_boundaries(scene_image, class_name=class_name)
                scene_image, self.x_shift, self.x_aspect_ratio, self.y_aspect_ratio = \
                    apply_crop_setting(scene_image, self.camera_type)

                if is_found_bucket:
                    image_info_dict = self.add_label(image_info_dict,
                                                     exported_image_name,
                                                     xmin, xmax, ymin, ymax,
                                                     class_name)
            if "BucketBB" in image_info_dict[exported_image_name] and \
                    "MatInside" in image_info_dict[exported_image_name]:
                image_info_dict = self.filter_matInside_outside_bucket(image_info_dict,
                                                                       exported_image_name)

            original_image_path = os.path.join(data_folder, "Scene", "Image",
                                               scene_image_name)
            original_image = cv2.imread(original_image_path)
            if original_image is None:
                original_image = cv2.imread(original_image_path.replace(".png",
                                                                        ".jpg"))

            if original_image is None:
                print('\n\n***********hs***********')
                print(original_image_path)
            else:
                original_image, self.x_shift, self.x_aspect_ratio, self.y_aspect_ratio = \
                    apply_crop_setting(original_image, self.camera_type)
                cv2.imwrite(os.path.join(save_path, exported_image_name), original_image)
                if self.is_debug:
                    image_debug = scene_image.copy()
                    image_debug = cv2.cvtColor(image_debug, cv2.COLOR_RGB2BGR)
                    self.visualize_img(image_debug, image_info_dict, exported_image_name,
                                       save_path, original_image=original_image)
        return image_info_dict

    def save_output(self):
        """ Iterates through each issue and then saves the resulting dict in
        a json file. """
        image_info_dict = {}
        folders, videos = self.read_csv

        print(folders)
        print(videos)

        for image_folder, video_path in zip(folders, videos):
            try:
                if self.label_type == "sequence_teeth":
                    image_folder = os.path.abspath(os.path.join(image_folder,
                                                                "../.."))
                    image_folder = os.path.join(image_folder,
                                                "SequenceTeeth", "Image")
                save_path = os.path.join(self.output_dir, "image")
                create_folder(save_path)
                if self.is_debug:
                    create_folder(os.path.join(save_path, "debug"))
                    create_folder(os.path.join(save_path, "debug_problematic"))
                is_wrong = False
                if "teeth" in self.label_type:  # NOTE!
                    root = self.parse_xml(os.path.join(image_folder, "Imageinfo.xml"))
                    image_info_dict, is_wrong = self.process_teeth(root,
                                                                   image_folder, save_path,
                                                                   image_info_dict,
                                                                   video_path,
                                                                   self.label_type)

                elif self.label_type == "scene":
                    image_info_dict = self.process_scene(image_folder, save_path,
                                                         image_info_dict)

                else:
                    raise Exception("Wrong args.label_type", self.label_type)
                if self.is_backhoe:
                    image_info_dict = self.process_scene(image_folder, save_path,
                                                         image_info_dict)

                print("Exporting the images of %s folder is done!" % image_folder)
                if is_wrong:
                    print("Issue %s wrong teeth coords!" % (image_folder))
            except Exception as e:
                print("======= Issue %s failed!\n %s =======" % (image_folder, e))
        with open(os.path.join(self.output_dir, "ImageInfo.json"), 'w') as f:
            json.dump(image_info_dict, f, indent=4, sort_keys=True)
        print("\nFinished putting images and json files of %d issues together!\n" % len(
            folders))


def correct_coords(xmin, ymin, xmax, ymax):
    if xmin < 0 and xmin > -7:
        xmin = 0
    if ymin < 0 and ymin > -7:
        ymin = 0
    return xmin, ymin, xmax, ymax


def assert_coords(xmin, ymin, xmax, ymax):
    if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0 or \
            xmin > xmax or ymin > ymax or \
            xmax > 730 or ymax > 490:
        print("Wrong Coordinates!", xmin, ymin, xmax, ymax)
        return False
    else:
        return True


def gen_xml_header(img_name, images_dir_path, video_path):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "image"
    ET.SubElement(annotation, "filename").text = img_name
    ET.SubElement(annotation, "path").text = os.path.join(images_dir_path, img_name)
    ET.SubElement(annotation, "video_path").text = video_path
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(640)
    ET.SubElement(size, "height").text = str(480)
    ET.SubElement(size, "depth").text = str(3)
    ET.SubElement(annotation, "segmented").text = str(0)
    return annotation


def gen_xml_files(args):
    input_dir = os.path.abspath(os.path.join(args.csv_file, os.pardir))
    labels_json_path = os.path.join(input_dir, "ImageInfo.json")
    images_dir_path = os.path.join(input_dir, "image")
    path_to_write_xml = os.path.join(args.output_dir, "labels")

    with open(labels_json_path) as f:
        data = json.load(f)
        labels_single_bbox = ["Toothline", "BucketBB", "MatInside", "WearArea"]
        labels_multiple_bbox = ["Tooth", "LipShroud"]

        for i, img in enumerate(data):
            if i % 100 == 0: print("Written %d/%d xml files/images." % (i, len(data)))
            annotation = gen_xml_header(img, images_dir_path, data[img]["VideoPath"])

            try:
                for label in labels_single_bbox:
                    if label not in data[img]: continue
                    if not data[img][label]: continue
                    object = ET.SubElement(annotation, "object")
                    ET.SubElement(object, "name").text = label
                    ET.SubElement(object, "truncated").text = str(0)
                    ET.SubElement(object, "difficult").text = str(0)
                    bndbox = ET.SubElement(object, "bndbox")
                    line_xmin = data[img][label]["X_CanvasLeft"]
                    line_ymin = data[img][label]["Y_CanvasTop"]
                    line_xmax = data[img][label]["X_CanvasLeft"] + \
                                data[img][label]["Width"]
                    line_ymax = data[img][label]["Y_CanvasTop"] + \
                                data[img][label]["Height"]
                    state = data[img][label]["state"]
                    ET.SubElement(bndbox, "xmin").text = str(line_xmin)
                    ET.SubElement(bndbox, "ymin").text = str(line_ymin)
                    ET.SubElement(bndbox, "xmax").text = str(line_xmax)
                    ET.SubElement(bndbox, "ymax").text = str(line_ymax)
                    if args.is_soft_state and (state == 1 or state == 0) and \
                            (label == "MatInside" or label == "WearArea"):
                        score = 0.5  # make it a soft label
                    else:
                        score = 1.
                    ET.SubElement(bndbox, "p").text = str(score)
                    if not assert_coords(line_xmin, line_ymin, line_xmax, line_ymax):
                        print("img %s wrong! %d %d %d %d" % \
                              (img, line_xmin, line_ymin, line_xmax, line_ymax))
                        annotation.remove(object)
            except Exception as e:
                print("Skipping Toothline/Bucket/WearArea due to error %s!" % e)

            for label in labels_multiple_bbox:
                if label in list(data[img].keys()):
                    for t in range(len(data[img][label])):
                        object = ET.SubElement(annotation, "object")
                        ET.SubElement(object, "name").text = label
                        ET.SubElement(object, "truncated").text = str(0)
                        ET.SubElement(object, "difficult").text = str(0)
                        bndbox = ET.SubElement(object, "bndbox")
                        xmin = data[img][label][t]["X_CanvasLeft"]
                        ymin = data[img][label][t]["Y_CanvasTop"]
                        xmax = data[img][label][t]["X_CanvasLeft"] + \
                               data[img][label][t]["Width"]
                        ymax = data[img][label][t]["Y_CanvasTop"] + \
                               data[img][label][t]["Height"]
                        # angle = data[img]["Tooth"][t]["rotate_angle"]

                        xmin, ymin, xmax, ymax = correct_coords(xmin, ymin, xmax, ymax)
                        if not assert_coords(xmin, ymin, xmax, ymax):
                            print("img %s wrong!" % img)
                            annotation.remove(object)
                            continue
                        else:
                            ET.SubElement(bndbox, "xmin").text = str(xmin)
                            ET.SubElement(bndbox, "ymin").text = str(ymin)
                            ET.SubElement(bndbox, "xmax").text = str(xmax)
                            ET.SubElement(bndbox, "ymax").text = str(ymax)
                            ET.SubElement(bndbox, "p").text = str(1.)

            tree = ET.ElementTree(annotation)
            new_xml_filepath = os.path.join(path_to_write_xml, img[:-3] + "xml")
            tree.write(new_xml_filepath)

    if args.background_dir:
        print("\nProcessing background images in %s." % args.background_dir)
        filenames = os.listdir(args.background_dir)
        for i, filename in enumerate(filenames):
            if i % 100 == 0: print("Written %d/%d background xml labels." % \
                                   (i, len(filenames)))
            filepath = os.path.join(args.background_dir, filename)
            image = cv2.imread(filepath)
            if image is not None:
                image, x_shift, x_aspect_ratio, y_aspect_ratio = \
                    apply_crop_setting(image, args.camera_type)
                image_name = filename.replace(".png", ".jpg")
                cv2.imwrite(os.path.join(images_dir_path, image_name), image)
                # NOTE: video_path is empty
                annotation = gen_xml_header(image_name, images_dir_path)
                tree = ET.ElementTree(annotation)
                new_xml_filepath = os.path.join(path_to_write_xml,
                                                image_name[:-3] + "xml")
                tree.write(new_xml_filepath)
            else:
                print("Image %s couldn't be read." % filename)

    print("\nFinished generating XML in VOC format!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", default="", type=str,
                        help="path to csv file exported from jira")
    parser.add_argument("--output-dir", default="", type=str,
                        help="path to save images and wear landmark labels")
    parser.add_argument("--camera-type", default="optical", type=str,
                        help="camera type for crop purpose")
    parser.add_argument("--background-dir", default="", type=str,
                        help="path to directory of background images")
    parser.add_argument("--label-type", type=str, default="teeth",
                        choices=["teeth", "sequence_teeth", "scene",
                                 "wear_fragmentation"],
                        help="Type of labeled image to process")
    parser.add_argument("--is-visible-part", action="store_true",
                        help="Whether to specify only visible height of the tooth instead"
                             "of the full height")
    parser.add_argument("--is-filter-visible", action="store_true",
                        help="Whether to exclude teeth which are partially visible")
    parser.add_argument("--is-filter-state", action="store_true",
                        help="Whether to exclude state=0 which are occluded teeth")
    parser.add_argument("--is-debug", action="store_true",
                        help="Debug mode would write images with bbox overlayed in the"
                             " debug folder")
    parser.add_argument("--is-backhoe", action="store_true",
                        help="If backhoe then bucket/matinside/weararea are processed from "
                             "scene labels and teeth bboxes are flipped vertically.")
    parser.add_argument("--is-soft-state", action="store_true",
                        help="Will assign a soft prob (<1.0) for yellow state=2")

    argv = vars(parser.parse_args())
    pp.pprint(argv)
    assert (argv['csv_file'] != "")
    assert (argv['csv_file'] != argv['output_dir'])

    argv = Initialization(**argv)
    if argv.label_type in ["teeth", "teethline", "scene", "sequence_teeth"]:
        image_info_dict = TeethlineTeeth(argv).save_output()

    if not os.path.exists(os.path.join(argv.output_dir, "labels")):
        os.makedirs(os.path.join(argv.output_dir, "labels"))

    print("\nGenerating individual xml file for each image...")
    gen_xml_files(argv)

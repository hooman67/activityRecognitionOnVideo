import os
import json
import csv
import argparse
import xml.etree.ElementTree
import pprint as pp
import math
from random import shuffle
import statistics

import cv2
import matplotlib.pyplot as plt

"""
Requires Python 3

NOTE: Might need to modify the N: path if running from Windows

This file imports a .csv file exported from Jira and produces a folder 'image' with
all images compiled and concatenated json file with all labels.
Saves keypoints in MPII format! It's also slightly modified in the original repo
of "Simple Baselines for Pose Estimation" paper.
"""

"""
python exportJiraImages.py --csv-file ... --output-dir ... --label-type keypoints --is-filter-state --is-debug
"""


def create_folder(directory):
    try:
        os.stat(directory)
    except FileNotFoundError:
        os.makedirs(directory)


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
            image = image[:, thermal_margin_start_width:
                             original_width + thermal_margin_start_width]
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
        self.camera_type = kwargs.get("camera_type")
        self.label_type = kwargs.get("label_type")
        self.is_filter_state = kwargs.get("is_filter_state")
        self.is_debug = kwargs.get("is_debug")
        self.is_soft_state = kwargs.get("is_soft_state")
        self.is_allow_bad_tooth_states = kwargs.get("is_allow_bad_tooth_states")

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
        assert value.lower() in ["keypoints", "wear", "fragmentation", "teeth",
                                 "teethline"],\
            "script does not support declared label type "
        self._label_type = value.lower()


class ReadCSVFile:
    def __init__(self, csv_file, label_type):
        self.csv_file = csv_file
        self.column_name = self.get_column_name(label_type)

    @staticmethod
    def get_column_name(label_type):
        if label_type in ["teethline", "teeth", "keypoints"]:
            column_name = "Teeth"
        else:
            column_name = "Scene"
        return column_name

    @property
    def read_csv(self):
        label_folders = []
        video_paths = []
        equipment_types = []
        with open(self.csv_file, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            first_row = next(reader)
            index_label_directory = first_row.index(
                "Custom field ({0} - Image Directory)".format(self.column_name))
            index_is_labeled = first_row.index(
                "Custom field ({0} Images Labeled)".format(self.column_name))
            index_video_path = first_row.index("Custom field (Full Path)")
            index_equipment_type = first_row.index("Custom field (Equipment Type/Model)")
            for row in reader:
                if row[index_is_labeled] == "Yes":
                    # label_folder = row[index_label_directory].replace('file:////motionmetrics.net/nas/', 'N:/')
                    label_folder = row[index_label_directory].replace(
                        'file:////motionmetrics.net/nas/', '/home/hooman/')
                    video_path = row[index_video_path].replace(
                        'file:////motionmetrics.net/nas/', '/home/hooman/')
                    equipment_type = row[index_equipment_type]
                    video_paths.append(video_path)
                    label_folders.append(os.path.normpath(label_folder))
                    equipment_types.append(equipment_type)
        return label_folders, video_paths, equipment_types


class TeethlineTeeth(ReadCSVFile):
    def __init__(self, args):
        self.label_type = args.label_type
        self.camera_type = args.camera_type
        self.output_dir = args.output_dir
        self.is_filter_state = args.is_filter_state
        self.x_shift = 0
        self.x_aspect_ratio = 1
        self.y_aspect_ratio = 1
        self.is_debug = args.is_debug
        self.is_soft_state = args.is_soft_state
        self.is_allow_bad_tooth_states = args.is_allow_bad_tooth_states
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
        if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0 or xmin > xmax or\
           ymin > ymax or xmax > 720 or ymax > 480: # or angle > 180 or angle < -180:
            print("Wrong Coordinates!", xmin, ymin, xmax, ymax, angle)
            return False
        else:
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

    def filter_state(self, info, excluded_states=[0]):
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

    def get_point_landmarks(self, info):
        """ In the order of y-axis, starting from the top
        - Lip Shroud
        - Lifting Eyes
        - Cast Lip
        - Bucket
        if element is None then the landmark is not labeled.
        """
        landmarks = [None, None, None, None]
        for landmark in info.findall('PointLandmarks'):
            landmark_name = landmark.find('Label').text
            landmark_x = int(round((float(landmark.find('X').text) - self.x_shift) * \
                                   self.x_aspect_ratio))
            landmark_y = int(round(float(landmark.find('Y').text) * \
                                   self.y_aspect_ratio))
            if landmark_name == "Lip Shroud":
                landmarks[0] = [landmark_x, landmark_y]
            elif landmark_name == "Lifting Eye":
                landmarks[1] = [landmark_x, landmark_y]
            elif landmark_name == "Cast Lip":
                landmarks[2] = [landmark_x, landmark_y]
            elif landmark_name == "Bucket":
                landmarks[3] = [landmark_x, landmark_y]
            else:
                print("Wrong landmark name: %s" % landmark_name)
                raise Exception

        return landmarks

    @staticmethod
    def get_center(keypoints):
        """ Gives a better center of a detection by:
        1) taking a middle of the range of values in x and y directions
        2) in case when only tooth tip or lip shroud is visible, take the center
        slightly more down so that it includes some of the wear area."""
        x_coords, y_coords = [], []
        for keypoint in keypoints:
            if keypoint is None: continue
            x_coords.append(keypoint[0])
            y_coords.append(keypoint[1])

        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        x_center = min(x_coords) + x_range / 2
        y_center = min(y_coords) + y_range / 2
        if len(y_coords) == 1:
            y_center += 30
        elif len(y_coords) == 2:
            y_center += 25

        return [x_center, y_center]

    def remove_keypoint_outliers(self, keypoints):
        """Remove keypoint if the keypoint is outside of image"""
        for i, keypoint in enumerate(keypoints):
            if keypoint is not None:
                if keypoint[0] < 0 or keypoint[0] > 720 or keypoint[1] < 0 or\
                   keypoint[1] > 480:
                    keypoints[i] = None

        return keypoints

    def remove_wrongly_assigned_keypoints(self, keypoints):
        """ Sometimes the landmarks are not correctly assigned to a tooth.
        (They are automatically assigned based on the distance to the tooth vertical)
        This function should remove landmarks if it's too far from the tooth_tip x-coord.
        """
        len_non_none = keypoints.count(None)
        if len_non_none <= 2:
            return keypoints

        tooth_tip = keypoints[0]
        if tooth_tip is None: return keypoints
        pixel_threshold = 30
        for i, keypoint in enumerate(keypoints[1:], start=1):  # skip tooth_tip
            if keypoint is None: continue
            delta = abs(keypoint[0] - tooth_tip[0])
            if delta > pixel_threshold:
                print("Removing an outlier landmark... ", delta)
                keypoints[i] = None

        return keypoints

    def get_incorrectly_assigned_keypoints(self, xml_image_info):
        """ Obtain landmarks which were incorrectly assigned to non-tooth
        bounding boxes.
        `landmarks` will be empty [] if there are no incorrectly assigned landmarks.
        """
        landmarks = []
        for info in xml_image_info.findall('Container'):
            if info.find("Label").text == "Tooth": continue

            landmarks = self.get_point_landmarks(info)
            len_non_none = len(landmarks) - landmarks.count(None)
            if len_non_none >= 2:
                break

        return landmarks

    def can_assignment_be_corrected(self, keypoints, landmarks_to_assigned):
        """ If any of the landmarks are within `pixel_threshold` to the
        tooth_tip then return True that this assignment may be correct."""
        tooth_tip = keypoints[0]
        if tooth_tip is None: return False

        is_assignment_to_be_corrected = False
        pixel_threshold = 12
        for landmark in landmarks_to_assigned:
            if landmark is None: continue
            if abs(landmark[0] - tooth_tip[0]) < pixel_threshold:
                is_assignment_to_be_corrected = True

        return is_assignment_to_be_corrected

    def correct_assignment(self, keypoints, landmarks_to_be_corrected):
        for i, landmark in enumerate(landmarks_to_be_corrected):
            if keypoints[i+1] is None:
                keypoints[i+1] = landmark

        return keypoints

    def compute_patch_scales(self, image_keypoints, eq_type):
        """ Computes x and y scale (multiple of 200 pixels).
        Width is computed as twice the median intertooth distance so that it
        doesn't include landmarks from neighbouring teeth. """
        tooth_tips_x = [keypoints[0][0] for keypoints in image_keypoints
                        if keypoints[0] is not None]
        inter_tooth_dists_x = [x2 - x1 for x1, x2 in
                               zip(tooth_tips_x[:-1], tooth_tips_x[1:])]
        if inter_tooth_dists_x:
            median_intertooth_dist_x = statistics.median(inter_tooth_dists_x)
        else:  # if no inter_tooth_dists, i.e. only a single tooth
            if eq_type == "Hydraulic":
                median_intertooth_dist_x = 60.
            elif eq_type == "Cable":
                median_intertooth_dist_x = 40.
            else:
                raise Exception("Add param for another equipment/shovel type!")

        pixel_std = 200.
        scale_x = 2 * median_intertooth_dist_x / pixel_std
        # assume patch height is around 3 time the width (worked well for cable)
        scale_y = 2 * 3 * median_intertooth_dist_x / pixel_std

        return scale_x, scale_y

    def extract_keypoints(self, mpii_json, xml_image_info, exported_image_name,
                          is_filter_state, eq_type):
        """ Extracts keypoints of:
            1) Tooth tip (taken from middle of two top points of teeth)
            2) Landmarks
        """
        eq_type = eq_type[10:].rstrip()
        incorrectly_assigned_keypoints =\
            self.get_incorrectly_assigned_keypoints(xml_image_info)
        image_keypoints = []
        start_index = len(mpii_json)
        for info in xml_image_info.findall('Container'):
            if info.find("Label").text != "Tooth": continue
            is_bad_state = False
            if is_filter_state:
                is_continue = self.filter_state(info, excluded_states=[0, 1])
                if is_continue: is_bad_state = True

            y_canvas_top = float(info.find("Y_CanvasTop").text)
            x_canvas_left = float(info.find("X_CanvasLeft").text)
            width = float(info.find("Width").text)
            height = float(info.find("Height").text)

            y_object_canvas_top = int(round(y_canvas_top * self.y_aspect_ratio))
            x_object_canvas_left = int(round(x_canvas_left - self.x_shift) *
                                       self.x_aspect_ratio)
            object_height = int(round(height * self.y_aspect_ratio))
            object_width = int(round(width * self.x_aspect_ratio))
            rotate_angle = int(info.find("RotateAngle").text)
            tooth_tip_x = x_object_canvas_left + (object_width) / 2
            tooth_tip_y = y_object_canvas_top

            point_landmarks = self.get_point_landmarks(info)

            if is_bad_state:
                if self.is_allow_bad_tooth_states:
                    tooth_tip = [tooth_tip_x, tooth_tip_y]
                else:
                    tooth_tip = None
            else:  # allow all state=2 tooth tips
                tooth_tip = [tooth_tip_x, tooth_tip_y]

            keypoints = [tooth_tip] + point_landmarks
            keypoints = self.remove_keypoint_outliers(keypoints)
            if self.can_assignment_be_corrected(keypoints,
                                                incorrectly_assigned_keypoints):
                keypoints = self.correct_assignment(keypoints,
                                                    incorrectly_assigned_keypoints)
            keypoints = self.remove_wrongly_assigned_keypoints(keypoints)
            is_wrong = False
            if is_wrong or (is_bad_state and not self.is_allow_bad_tooth_states) or\
               keypoints.count(None) == len(keypoints):
                    continue

            image_keypoints.append(keypoints)
            # we want smaller areas for cable shovels and larger for hydraulic
            if eq_type == "Bucyrus" or eq_type == "P&H":
                scale = 0.6  # multiple of 200 pixels
            elif eq_type == "Hydraulic":  # Hydraulic
                scale = 0.8
            elif eq_type == "Backhoe":
                scale = 0.7
            else:
                raise Exception("Non-supported shovel type!!!")

            label_to_add = {
                "joints_vis":
                    [1 if keypoint is not None else 0 for keypoint in keypoints],
                "joints": [keypoint if keypoint is not None else [-1., -1.]
                           for keypoint in keypoints],
                "image": exported_image_name,
                "scale": scale,
                "center": self.get_center(keypoints)
            }
            mpii_json.append(label_to_add)


        if image_keypoints == []:
            is_whole_image_wrong = True
        else:
            is_whole_image_wrong = False

            end_index = len(mpii_json)
            scale_x, scale_y = self.compute_patch_scales(image_keypoints,
                                                         eq_type)
            for i_mpii in range(start_index, end_index):
                mpii_json[i_mpii]['scale_x'] = scale_x
                mpii_json[i_mpii]['scale_y'] = scale_y

        return mpii_json, is_whole_image_wrong, image_keypoints

    def visualize_img_keypoints(self, image, image_keypoints, exported_image_name,
                                save_path, folder_name='debug'):
        """ Visualizes the keypoints and saves in `folder_name`.
        Keep in mind, it doesn't show the assignment of each keypoint to a detection"""
        plt.figure(figsize=(12, 9))
        plt.imshow(image)
        colors = ["purple", "green", "red", "blue", "yellow"]
        for tooth_patch in image_keypoints:
            for i, keypoint in enumerate(tooth_patch):
                if keypoint is None or int(keypoint[0]) == -1:
                    continue

                plt.scatter(keypoint[0], keypoint[1], s=30, c=colors[i], marker='o',
                            alpha=0.7)

        plt.title(exported_image_name)
        # plt.show()
        path = os.path.join(save_path, folder_name, exported_image_name)
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight', pad_inches=0, format='jpg')
        plt.close()

    def process_teeth_and_wear(self, root, mpii_json, image_folder, save_path, eq_type):
        is_wrong = False
        for xml_image in root.findall('XMLSaveThumbnail'):
            original_image_name = xml_image.get("Path")
            exported_image_name = original_image_name.replace(".png", ".jpg")
            image = cv2.imread(os.path.join(image_folder, original_image_name))
            if image is None:
                image = cv2.imread(os.path.join(image_folder,
                                                original_image_name.replace(".png",
                                                                            ".jpg")))
            image, self.x_shift, self.x_aspect_ratio, self.y_aspect_ratio = \
                apply_crop_setting(image, self.camera_type)
            mpii_json, is_wrong, image_keypoints = \
                self.extract_keypoints(mpii_json, xml_image, exported_image_name,
                                       self.is_filter_state, eq_type)

            has_container = xml_image.find("HasContainer").text
            is_wrong = is_wrong or not has_container

            if not is_wrong:
                cv2.imwrite(os.path.join(save_path, exported_image_name), image)

                if self.is_debug:
                    image_debug = image.copy()
                    self.visualize_img_keypoints(image_debug, image_keypoints,
                                                 exported_image_name, save_path)

        return mpii_json, is_wrong

    def save_dataset(self, mpii_json, val_ratio=0.08):
        shuffle(mpii_json)
        num_samples = len(mpii_json)
        train_val_boundary = int(num_samples * (1. - val_ratio))
        train_set = mpii_json[:train_val_boundary]
        valid_set = mpii_json[train_val_boundary:]

        with open(os.path.join(self.output_dir, "annot", "train.json"), 'w') as f:
            json.dump(train_set, f, indent=4, sort_keys=True)
        with open(os.path.join(self.output_dir, "annot", "valid.json"), 'w') as f:
            json.dump(valid_set, f, indent=4, sort_keys=True)
        return

    def save_output(self):
        """ Iterates through each issue and then saves the resulting dict in
        a json file. """
        mpii_json = []
        folders, videos, eq_types = self.read_csv
        for image_folder, video_path, eq_type in zip(folders, videos, eq_types):
            try:
                save_path = os.path.join(self.output_dir, "images")
                save_path_annot = os.path.join(self.output_dir, "annot")
                create_folder(save_path)
                create_folder(save_path_annot)
                if self.is_debug:
                    create_folder(os.path.join(save_path, "debug"))
                is_wrong = False
                if "keypoints" in self.label_type:  # NOTE!
                    root = self.parse_xml(os.path.join(image_folder, "Imageinfo.xml"))
                    mpii_json, is_wrong = self.process_teeth_and_wear(root,
                                                                      mpii_json,
                                                                      image_folder,
                                                                      save_path,
                                                                      eq_type)
                else:
                    raise Exception("Wrong args.label_type", self.label_type)

                print("Exporting the images of %s folder is done!" % image_folder)
                if is_wrong:
                    print("Issue %s wrong teeth coords!" % (image_folder))
            except Exception as e:
                print("======= Issue %s failed!\n %s =======" % (image_folder, e))
        self.save_dataset(mpii_json, val_ratio=0.08)
        print("\nFinished putting images and json files of %d issues together!\n" %
              len(folders))


def correct_coords(xmin, ymin, xmax, ymax):
    if xmin < 0 and xmin > -7:
        xmin = 0
    if ymin < 0 and ymin > -7:
        ymin = 0

    return xmin, ymin, xmax, ymax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", default="", type=str, required=True,
                        help="path to csv file exported from jira")
    parser.add_argument("--output-dir", default="", type=str, required=True,
                        help="path to save images and wear landmark labels")
    parser.add_argument("--camera-type", default="optical", type=str,
                        help="camera type for crop purpose")
    parser.add_argument("--label-type", type=str, default="keypoints",
                        choices=["keypoints"],
                        help="Type of labeled image to process")
    parser.add_argument("--is-filter-state", action="store_true",
                        help="Whether to exclude state=0 which are occluded teeth")
    parser.add_argument("--is-debug", action="store_true",
                        help="Debug mode would write images with bbox overlayed in the"
                             " debug folder")
    parser.add_argument("--is-allow-bad-tooth-states", action="store_true",
                        help="Include patches of state zero or one. ")

    argv = vars(parser.parse_args())
    pp.pprint(argv)
    assert (argv['csv_file'] != "")
    assert (argv['csv_file'] != argv['output_dir'])

    argv = Initialization(**argv)
    if argv.label_type in ["keypoints"]:
        image_info_dict = TeethlineTeeth(argv).save_output()

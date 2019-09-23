import json
import xml.etree.cElementTree as ET
import os
import math
import argparse
import numpy as np
import cv2
import pprint as pp
from copy import copy
import sys

"""
Creates an xml file for each image in Pascal VOC format, takes an input from
exportJiraImages.
Also computes statistics (quantiles) of labels.

VOC format example:
<annotation>
	<folder>GeneratedData_Train</folder>
	<filename>000001.png</filename>
	<path>/my/path/GeneratedData_Train/000001.png</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>224</width>
		<height>224</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>21</name>
		<pose>Frontal</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<occluded>0</occluded>
		<bndbox>
			<xmin>82</xmin>
			<xmax>172</xmax>
			<ymin>88</ymin>
			<ymax>146</ymax>
		</bndbox>
	</object>
</annotation>


Additionally we have `video_path` field under `annotation`.
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="", type=str, help="path to images and .json labels")
    parser.add_argument("--output_dir", default="", type=str, help="usually the same as input-dir")
    parser.add_argument("--is_toothline", action="store_true")

    args = parser.parse_args()
    return args

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except:
        return

def correct_coords(xmin, ymin, xmax, ymax):
    if xmin < 0 and xmin > -7:
        xmin = 0
    if ymin < 0 and ymin > -7:
        ymin = 0

    return xmin, ymin, xmax, ymax


def assert_coords(xmin, ymin, xmax, ymax):
    if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0 or\
            xmin > xmax or ymin > ymax or\
            xmax > 730 or ymax > 490:
        print "Wrong Coordinates!", xmin, ymin, xmax, ymax
        return False
    else:
        return True


def instantiate_xml_file(data, img_name):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation,"folder").text = "image"
    ET.SubElement(annotation,"filename").text = img_name
    ET.SubElement(annotation, "path").text = os.path.join(images_dir_path, img_name)
    ET.SubElement(annotation, "video_path").text = data[img_name]["VideoPath"]
    size = ET.SubElement(annotation,"size")
    ET.SubElement(size,"width").text = str(640)
    ET.SubElement(size,"height").text = str(480)
    ET.SubElement(size,"depth").text = str(3)
    ET.SubElement(annotation, "segmented").text = str(0)	
    return annotation


def add_single_labels(data, img_name, annotation):
    try: 
        for label in labels_single_bbox:
            if label not in data[img_name]: continue
            if not data[img_name][label]: continue
            object = ET.SubElement(annotation, "object")
            ET.SubElement(object, "name").text = label
            ET.SubElement(object, "truncated").text = str(0)
            ET.SubElement(object, "difficult").text = str(0)
            bndbox = ET.SubElement(object,"bndbox")
            line_xmin = data[img_name][label]["X_CanvasLeft"]
            line_ymin = data[img_name][label]["Y_CanvasTop"]
            line_xmax = data[img_name][label]["X_CanvasLeft"] +\
                        data[img_name][label]["Width"]
            line_ymax = data[img_name][label]["Y_CanvasTop"] +\
                        data[img_name][label]["Height"]
            ET.SubElement(bndbox, "xmin").text = str(line_xmin)
            ET.SubElement(bndbox, "ymin").text = str(line_ymin)
            ET.SubElement(bndbox, "xmax").text = str(line_xmax)
            ET.SubElement(bndbox, "ymax").text = str(line_ymax)
            if not assert_coords(line_xmin, line_ymin, line_xmax, line_ymax): 
                print "img %s wrong!" % img_name
                annotation.remove(object)
    except:
        print "Skipping Toothline/Bucket/WearArea due to error!"
    return annotation


def add_plural_labels(data, img_name, annotation):
    for class_name in ["Bucket", "MatInsideBucket", "WearArea"]:
        if class_name not in data[img_name].keys(): continue
        object = ET.SubElement(annotation,"object")
        ET.SubElement(object,"name").text = class_name 
        ET.SubElement(object, "truncated").text = str(0)
        ET.SubElement(object, "difficult").text = str(0)
        bndbox = ET.SubElement(object,"bndbox")
        line_xmin = data[img_name][class_name]["X_CanvasLeft"]
        line_ymin = data[img_name][class_name]["Y_CanvasTop"]
        line_xmax = data[img_name][class_name]["X_CanvasLeft"] +\
                data[img_name][class_name]["Width"]
        line_ymax = data[img_name][class_name]["Y_CanvasTop"] +\
                data[img_name][class_name]["Height"]
        ET.SubElement(bndbox,"xmin").text = str(line_xmin)
        ET.SubElement(bndbox,"ymin").text = str(line_ymin)
        ET.SubElement(bndbox,"xmax").text = str(line_xmax)
        ET.SubElement(bndbox,"ymax").text = str(line_ymax)
        if not assert_coords(line_xmin, line_ymin, line_xmax, line_ymax): 
            print "img %s wrong!" % img_name
            annotation.remove(object)


def gen_xml_labels(args):
    labels_json_path = os.path.join(args['input_dir'], "ImageInfo.json")
    images_dir_path = os.path.join(args['input_dir'], "image")
    path_to_write_xml = os.path.join(args['output_dir'], "labels")

    with open(labels_json_path) as f:
        data = json.load(f)
        labels_single_bbox = ["Toothline", "BucketBB", "MatInside", "WearArea"]
        labels_multiple_bbox = ["Tooth", "LipShroud"]

        heights = []
        widths = []
        heights_toothline = []
        widths_toothline = []
        for i, img in enumerate(data):
            if i % 100 == 0: print("Written %d/%d xml files/images." % (i, len(data)))
            annotation = ET.Element("annotation")
            ET.SubElement(annotation,"folder").text = "image"
            ET.SubElement(annotation,"filename").text = img
            ET.SubElement(annotation, "path").text = os.path.join(images_dir_path, img)
            ET.SubElement(annotation, "video_path").text = data[img]["VideoPath"]
            size = ET.SubElement(annotation,"size")
            ET.SubElement(size,"width").text = str(640)
            ET.SubElement(size,"height").text = str(480)
            ET.SubElement(size,"depth").text = str(3)
            ET.SubElement(annotation, "segmented").text = str(0)	

            if args['is_toothline']:
                try: 
                    for label in labels_single_bbox:
                        if label not in data[img]: continue
                        if not data[img][label]: continue
                        object = ET.SubElement(annotation, "object")
                        ET.SubElement(object, "name").text = label
                        ET.SubElement(object, "truncated").text = str(0)
                        ET.SubElement(object, "difficult").text = str(0)
                        bndbox = ET.SubElement(object,"bndbox")
                        line_xmin = data[img][label]["X_CanvasLeft"]
                        line_ymin = data[img][label]["Y_CanvasTop"]
                        line_xmax = data[img][label]["X_CanvasLeft"] +\
                                    data[img][label]["Width"]
                        line_ymax = data[img][label]["Y_CanvasTop"] +\
                                    data[img][label]["Height"]
                        ET.SubElement(bndbox, "xmin").text = str(line_xmin)
                        ET.SubElement(bndbox, "ymin").text = str(line_ymin)
                        ET.SubElement(bndbox, "xmax").text = str(line_xmax)
                        ET.SubElement(bndbox, "ymax").text = str(line_ymax)
                        if not assert_coords(line_xmin, line_ymin, line_xmax, line_ymax): 
                            print "img %s wrong!" % img
                            annotation.remove(object)
                except:
                    print "Skipping Toothline/Bucket/WearArea due to error!"

            for label in labels_multiple_bbox:
                if label in data[img].keys():
                    for t in range(len(data[img][label])):
                        object = ET.SubElement(annotation, "object")
                        ET.SubElement(object, "name").text = label
                        ET.SubElement(object, "truncated").text = str(0)
                        ET.SubElement(object, "difficult").text = str(0)
                        bndbox = ET.SubElement(object,"bndbox")
                        xmin = data[img][label][t]["X_CanvasLeft"]
                        ymin = data[img][label][t]["Y_CanvasTop"]
                        xmax = data[img][label][t]["X_CanvasLeft"] +\
                               data[img][label][t]["Width"]
                        ymax = data[img][label][t]["Y_CanvasTop"] +\
                               data[img][label][t]["Height"]
                        # angle = data[img]["Tooth"][t]["rotate_angle"]
                        if xmax - xmin < 10:
                            xmin -= 2
                            xmax += 2
                        elif xmax - xmin < 15 and xmax - xmin >= 10:
                            xmin -= 1
                            xmax += 1
                        if ymax - ymin < 10:
                            ymin -= 2
                            ymax += 2
                        elif ymax - ymin < 15 and ymax - ymin >= 10:
                            ymin -= 1
                            ymax += 1

                        xmin, ymin, xmax, ymax = correct_coords(xmin, ymin, xmax, ymax)
                        if not assert_coords(xmin, ymin, xmax, ymax): 
                            print "img %s wrong!" % img
                            annotation.remove(object)
                            continue
                        else:
                            ET.SubElement(bndbox,"xmin").text = str(xmin)
                            ET.SubElement(bndbox,"ymin").text = str(ymin)
                            ET.SubElement(bndbox,"xmax").text = str(xmax)
                            ET.SubElement(bndbox,"ymax").text = str(ymax)

                        heights.append(abs(ymax - ymin))
                        widths.append(abs(xmax - xmin))

            for class_name in ["Bucket", "MatInsideBucket", "WearArea"]:
                if class_name not in data[img].keys(): continue
                object = ET.SubElement(annotation,"object")
                ET.SubElement(object,"name").text = class_name 
                ET.SubElement(object, "truncated").text = str(0)
                ET.SubElement(object, "difficult").text = str(0)
                bndbox = ET.SubElement(object,"bndbox")
                line_xmin = data[img][class_name]["X_CanvasLeft"]
                line_ymin = data[img][class_name]["Y_CanvasTop"]
                line_xmax = data[img][class_name]["X_CanvasLeft"] +\
                        data[img][class_name]["Width"]
                line_ymax = data[img][class_name]["Y_CanvasTop"] +\
                        data[img][class_name]["Height"]
                ET.SubElement(bndbox,"xmin").text = str(line_xmin)
                ET.SubElement(bndbox,"ymin").text = str(line_ymin)
                ET.SubElement(bndbox,"xmax").text = str(line_xmax)
                ET.SubElement(bndbox,"ymax").text = str(line_ymax)
                if not assert_coords(line_xmin, line_ymin, line_xmax, line_ymax): 
                    print "img %s wrong!" % img
                    annotation.remove(object)

            tree = ET.ElementTree(annotation)
            new_xml_filepath = os.path.join(path_to_write_xml, img[:-3] + "xml")
            tree.write(new_xml_filepath)

    print "\nFinished generating XML in VOC format!\n"

    # heights = np.array(heights)
    # widths = np.array(widths)
    # print "Height min/max: ", np.min(heights), np.max(heights)
    # print "Width min/max: ", np.min(heights), np.max(widths)
    # print "Quantiles height: ", np.percentile(heights, 10), np.percentile(heights, 50), np.percentile(heights, 90)
    # print "Quantiles widths: ", np.percentile(widths, 10), np.percentile(widths, 50), np.percentile(widths, 90)
    # print "\nToothline:\nHeight min/max: ", np.min(heights_toothline), np.max(heights_toothline)
    # print "Width min/max: ", np.min(heights_toothline), np.max(widths_toothline)
    # print "Quantiles height: ", np.percentile(heights_toothline, 10), np.percentile(heights_toothline, 50), np.percentile(heights_toothline, 90)
    # print "Quantiles widths: ", np.percentile(widths_toothline, 10), np.percentile(widths_toothline, 50), np.percentile(widths_toothline, 90)



def associate_wrt_toothline(toothline, tooth_bbox, teeth_tracks, num_teeth, i_tooth,
                            img_names, img_name):
    """ Simple association from detections to specific tooth by partitioning 
    toothline into num_teeth number of areas """
    center_x = tooth_bbox[0] + tooth_bbox[2] / 2
    center_y = tooth_bbox[1] + tooth_bbox[3] / 2
    for i in range(num_teeth):
        tooth_toothline_position_x_left  = toothline[0] +\
                (toothline[2] * i / num_teeth) - 3
        tooth_toothline_position_x_right = toothline[0] +\
                (toothline[2] * (i+1) / num_teeth) + 3
        tooth_toothline_position_y_up    = toothline[1] - 3
        tooth_toothline_position_y_down  = toothline[1] + toothline[3] + 3
        if center_x >= tooth_toothline_position_x_left and\
                center_x <= tooth_toothline_position_x_right and\
                center_y >= tooth_toothline_position_y_up and\
                center_y <= tooth_toothline_position_y_down:

            teeth_tracks[i][i_tooth] = tooth_bbox
            img_names[i][i_tooth] = img_name
            return teeth_tracks, img_names
    print "TOOTH NOT ASSOCIATED!"
    return teeth_tracks, img_names


if __name__ == "__main__":
    args = parse_args()
    args = vars(args)
    pp.pprint(args)

    if not os.path.exists(os.path.join(args['output_dir'], "labels")):
        os.makedirs(os.path.join(args['output_dir'], "labels"))

    gen_xml_labels(args)
    

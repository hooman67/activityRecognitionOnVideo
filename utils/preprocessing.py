import os
import sys
import copy
import xml.etree.ElementTree as ET
import math
from collections import Counter

import pprint as pp
import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from keras.utils import Sequence
import matplotlib.pyplot as plt

sys.path.append('../')
from utils.utils import bbox_iou, BoundBox
from utils.timestamp import process_xml_to_dict


def split_into_ranges(all_anns, sequence_length=30, stride=30, give_dataset_info=True):
    """ sequence_length of each sample, stride is to skip a number of frames
    in order to reduce redundancy of input data.
    Only one sample is taken from each "Still" segment because visually the
    network would remember the same samples instead remembering the no-motion
    state. """
    counter_samples = Counter()
    counter_transitions = Counter()
    splitted_anns = []
    for issue_dict in all_anns:
        video_path = issue_dict["video_path"]
        activity_anns = issue_dict["activities"]
        all_ranges = []
        issue_splitted_anns = []
        for activity, frame_ranges in activity_anns.items():
            all_ranges += frame_ranges
            for frame_range in frame_ranges:
                begin, end = frame_range
                num_of_samples = (end - begin) / (sequence_length + stride)
                for i in range(math.floor(num_of_samples)):
                    if activity == "Still": i += 1  # +1 so that it's not on the edge
                    sample = {"video_path": video_path,
                              "activity": activity,
                              "frame_number": begin + i * (sequence_length+stride)}
                    splitted_anns.append(sample)
                    issue_splitted_anns.append(sample)
                    counter_samples[activity] += 1

                    if activity == "Still":
                        break

        """ Due to the fact that some time segments are not labeled (assumed to be
        background), we fill it as a 'Swing' label 
        (perhaps better use another class?!). """
        activity_to_fill = "Background"
        all_ranges_sorted = sorted(all_ranges, key=lambda x: x[0])
        for range1, range2 in zip(all_ranges_sorted[:-1], all_ranges_sorted[1:]):
            diff_frames = range2[0] - range1[1]
            num_of_samples = diff_frames / (sequence_length + stride)
            if num_of_samples <= 2 or num_of_samples >= 6:
                continue
            for i in range(math.floor(num_of_samples)):
                sample = {"video_path": video_path,
                          "activity": activity_to_fill,
                          "frame_number": range1[1] + i*(sequence_length+stride)}
                splitted_anns.append(sample)
                issue_splitted_anns.append(sample)
                counter_samples[activity_to_fill] += 1

        if give_dataset_info:
            sorted_anns = sorted(issue_splitted_anns, key=lambda k: k['frame_number'])
            prev_act = sorted_anns[0]['activity']
            for sample in sorted_anns[1:]:
                curr_act = sample['activity']
                if prev_act != curr_act:
                    transition = prev_act + "->" + curr_act
                    transition = transition.replace("Swing", "Swing Full")
                    transition = transition.replace("Background", "Swing Empty")
                    counter_transitions[transition] += 1
                    prev_act = curr_act

    if give_dataset_info:
        pp.pprint(counter_samples)
        pp.pprint(counter_transitions)
        num_trans = sum([val for val in counter_transitions.values()])
        print(num_trans)
        counter_transitions_perc = {key: 100.*val/num_trans for key, val in\
                                    counter_transitions.items()}

    return splitted_anns


def augment_with_act(splitted_anns, ratio_of_still=0.15, act_to_fill="Still"):
    """ This is done to increase the number of Still/Idle examples because we can
    only take one sequence from the whole single Still period in a video.
    Keep in mind that in the generator the Still activity has to be explicitly
    caught so that extracted frames actually have no motion. """
    every_nth_sample = math.ceil(1. / ratio_of_still)
    for i in range(0, len(splitted_anns), every_nth_sample):
        sample = splitted_anns[i]
        new_sample = sample.copy()
        new_sample['activity'] = act_to_fill
        # let the third of examples be existing labels but without motion
        if np.random.random() > 0.33:
            frame_rate = 30
            new_sample['frame_number'] += 0.5 * frame_rate  # 0.5 is hard-coded
        splitted_anns.append(new_sample)

    return splitted_anns


def parse_annotation_activity(activity_folder):
    print("\nStarting annotation parsing...")
    all_anns = []

    for ann in sorted(os.listdir(activity_folder)):
        if ann.endswith("xml"):
            issue_path = os.path.join(activity_folder, ann)
            activity_dict = process_xml_to_dict(issue_path)
            all_anns.append(activity_dict)

    splitted_anns = split_into_ranges(all_anns)
    splitted_anns = augment_with_act(splitted_anns)

    return splitted_anns


def parse_annotation(ann_dir, img_dir, labels=[]):
    print("\nStarting annotation parsing...")
    all_imgs = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        img = {'object': []}

        try:
            tree = ET.parse(os.path.join(ann_dir, ann))
        except Exception as e:
            print("Skipping image %s due to:\n%s" % (ann, e))
            continue

        for elem in tree.iter():
            if 'filename' in elem.tag:
                # img['filename'] = img_dir + elem.text
                img['filename'] = os.path.join(img_dir, elem.text)
            if 'video_path' in elem.tag:
                img['video_path'] = elem.text
            if 'activity_label' in elem.tag:
                img['activity_label'] = elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))
                            if 'p' in dim.tag:
                                obj['p'] = float(dim.text)
                        if 'p' not in obj:
                            obj['p'] = 1.

        if not img['object']:
            img['object'] = {}

        # if len(img['object']) > 0:  # exclude images without objects
        if len(img['object']) >= 0:  # include images without objects
            all_imgs += [img]

    print("seen_labels: \n")
    pp.pprint(seen_labels)
    print("Finished reading annotations!")

    return all_imgs, seen_labels


class BatchGenerator(Sequence):
    def __init__(self, images,
                 config,
                 shuffle=True,
                 jitter=True,
                 norm=None,
                 debug=False):
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm
        self.debug = debug

        self.anchors = [
            BoundBox(0, 0, config['ANCHORS'][2 * i], config['ANCHORS'][2 * i + 1]) for i
            in range(int(len(config['ANCHORS']) // 2))]

        ### augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                # iaa.Fliplr(0.5), # horizontally flip 50% of all images
                # iaa.Flipud(0.2), # vertically flip 20% of all images
                # sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                # sometimes(iaa.Affine(
                # scale={"x": (0.85, 1.15), "y": (0.85, 1.15)}, # scale images to 80-120% of their size, individually per axis
                # translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, # translate by -20 to +20 percent (per axis)
                # rotate=(-5, 5), # rotate by -45 to +45 degrees
                # shear=(-5, 5), # shear by -16 to +16 degrees
                # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                # )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [
                               # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 2.0)),
                                   # blur images with a sigma between 0 and 3.0
                                   # iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                                   # iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                               # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                               # search either for all edges or for directed edges
                               # sometimes(iaa.OneOf([
                               #    iaa.EdgeDetect(alpha=(0, 0.7)),
                               #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                               # ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.04 * 255),
                                                         per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.05), per_channel=0.5),
                                   # randomly remove up to 10% of the pixels
                                   # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               # iaa.Invert(0.05, per_channel=True), # invert color channels
                               # iaa.Add((-6, 6), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                               iaa.Multiply((0.8, 1.25), per_channel=0.5),
                               # change brightness of images (50-150% of original value)
                               iaa.ContrastNormalization((0.8, 1.25), per_channel=0.5),
                               # improve or worsen the contrast
                               # iaa.Grayscale(alpha=(0.0, 1.0)),
                               # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                               # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

        if shuffle: np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images)) / self.config['BATCH_SIZE']))

    def num_classes(self):
        return len(self.config['LABELS'])

    def size(self):
        return len(self.images)

    def load_annotation(self, i):
        annots = []

        for obj in self.images[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'],
                     self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0:
            annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.images[i]['filename'])

    def __getitem__(self, idx):
        l_bound = idx * self.config['BATCH_SIZE']
        r_bound = (idx + 1) * self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'],
                            self.config['IMAGE_W'], 3))  # input images
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.config['TRUE_BOX_BUFFER'],
                            4))  # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],
                            self.config['GRID_W'], self.config['BOX'],
                            4 + 1 + len(self.config['LABELS'])))  # desired network output

        for train_instance in self.images[l_bound:r_bound]:
            img, all_objs = self.aug_image(train_instance, jitter=self.jitter)

            # construct output from object's x, y, w, h
            true_box_index = 0
            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj[
                    'name'] in self.config['LABELS']:
                    center_x = .5 * (obj['xmin'] + obj['xmax'])
                    center_x = center_x / (
                                float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    center_y = .5 * (obj['ymin'] + obj['ymax'])
                    center_y = center_y / (
                                float(self.config['IMAGE_H']) / self.config['GRID_H'])

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                        obj_indx = self.config['LABELS'].index(obj['name'])

                        center_w = (obj['xmax'] - obj['xmin']) / (
                                    float(self.config['IMAGE_W']) / self.config[
                                'GRID_W'])  # unit: grid cell
                        center_h = (obj['ymax'] - obj['ymin']) / (
                                    float(self.config['IMAGE_H']) / self.config[
                                'GRID_H'])  # unit: grid cell

                        box = [center_x, center_y, center_w, center_h]

                        # find the anchor that best predicts this box
                        best_anchor = -1
                        max_iou = -1

                        shifted_box = BoundBox(0,
                                               0,
                                               center_w,
                                               center_h)

                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou = bbox_iou(shifted_box, anchor)

                            if max_iou < iou:
                                best_anchor = i
                                max_iou = iou

                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4] = obj['p']
                        y_batch[
                            instance_count, grid_y, grid_x, best_anchor, 5 + obj_indx] = 1

                        # assign the true box to b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box

                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

            # assign input image to x_batch
            if self.norm != None:
                x_batch[instance_count] = self.norm(img)
            if self.debug:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                        cv2.rectangle(img[:, :, ::-1], (obj['xmin'], obj['ymin']),
                                      (obj['xmax'], obj['ymax']), (255, 0, 0), 3)
                        cv2.putText(img[:, :, ::-1], obj['name'],
                                    (obj['xmin'] + 2, obj['ymin'] + 12),
                                    0, 1.2e-3 * img.shape[0],
                                    (0, 255, 0), 1)
                plt.figure(figsize=(12, 10))
                plt.imshow(img)
                plt.title("Testing")
                plt.show()

                x_batch[instance_count] = img

            # increase instance counter in current batch
            instance_count += 1

            # print(' new batch created', idx)

        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']
        image = cv2.imread(image_name)

        if image is None:
            print('Cannot find ', image_name)
            print("Trying again...")
            image = cv2.imread(image_name)
            if image is None:
                print("Still cannot find image")
                raise ValueError()

        h, w, c = image.shape
        all_objs = copy.deepcopy(train_instance['object'])

        if jitter:
            ### scale the image
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

            ### translate the image
            max_offx = (scale - 1.) * w
            max_offy = (scale - 1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)

            image = image[offy: (offy + h), offx: (offx + w)]

            ### flip the image
            flip = np.random.binomial(1, .5)
            if flip > 0.5: image = cv2.flip(image, 1)

            image = self.aug_pipe.augment_image(image)

            # resize the image to standard size
        image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))
        image = image[:, :, ::-1]

        # fix object's position and size
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offx)

                obj[attr] = int(obj[attr] * float(self.config['IMAGE_W']) / w)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)

            for attr in ['ymin', 'ymax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offy)

                obj[attr] = int(obj[attr] * float(self.config['IMAGE_H']) / h)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)

            if jitter and flip > 0.5:
                xmin = obj['xmin']
                obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
                obj['xmax'] = self.config['IMAGE_W'] - xmin

        return image, all_objs



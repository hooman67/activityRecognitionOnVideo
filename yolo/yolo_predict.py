import os
import sys
import argparse
import json
import pprint as pp

from tqdm import tqdm
import cv2
import numpy as np
from keras import backend as K
import tensorflow as tf

sys.path.append('../')
from yolo.frontend import YOLO
from utils.xml_utils import gen_xml_file
from utils.preprocessing import parse_annotation, BatchGenerator
from utils.utils import *


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

obj_thresh = 0.2
class_obj_threshold = [0.4, 0.3, 0.3, 0.3, 0.3, 0.3]

shovel_type = "Cable"
if shovel_type == "Cable": num_teeth = 8
elif shovel_type == "Hydraulic": num_teeth = 6
elif shovel_type == "Backhoe": num_teeth = 6


argparser = argparse.ArgumentParser(
    description='Train and validate yolo backbone on any dataset')
argparser.add_argument('-c', '--conf', required=True, help='path to configuration file')
argparser.add_argument('-w', '--weights', required=True, help='path to pretrained weights')
argparser.add_argument('-i', '--input', required=True,
                       help='path to an image or a video (mp4 format)')
argparser.add_argument('-o', '--output', type=str, required=True,
                       help='path to save images/video.')
argparser.add_argument('-s', '--save_soft', action='store_true',
                       help='whether to save soft predicted labels in xml file (only for images)')
argparser.add_argument('-e', '--evaluate',
                       help='whether to evaluated on a labeled dataset or not')


def predictOnVideo(video_path, output_path, config, maxFramesToRunOn, detectionModel):    
    
    video_out_name = config['model']['backend'] + "_" +\
                     video_path.split('/')[-1][:-4] + ".avi"
    video_out = os.path.join(output_path, video_out_name)
    
    video_reader = cv2.VideoCapture(video_path)
    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    frames_to_predict = min(maxFramesToRunOn, nb_frames-30)
    skip_rate = 3
    frame_rate = 30
    video_writer = cv2.VideoWriter(video_out,
                           cv2.VideoWriter_fourcc(*'MPEG'), 
                           frame_rate / skip_rate, 
                           (frame_w, frame_h))

    bboxes_in_all_frames = []
    for i in tqdm(list(range(frames_to_predict))):
        _, image = video_reader.read()

        if i % skip_rate == 0:
            start_time = time()
            boxes, filtered_boxes = detectionModel.predict(
                image, 
                obj_threshold=obj_thresh,
                nms_threshold=0.01,
                is_filter_bboxes=False,
                shovel_type=shovel_type,
                class_obj_threshold=class_obj_threshold
            )
            print("Total Time: %.3f" % (time() - start_time))
          
            #we do this to draw both original and filtered boxes. Draw boxes knows to draw them with different thicknesses.
            boxes += filtered_boxes
                     
            image = draw_boxes(
                image,
                boxes,
               config['model']['labels'],
               score_threshold=obj_thresh,
               class_obj_threshold=class_obj_threshold
            )

            video_writer.write(np.uint8(image))
            bboxes_in_all_frames.append(boxes)

    video_reader.release()
    video_writer.release()  


def predictOnImageDir(image_dir_path, output_path, config,savePredictionsAsXmlToo, detectionModel):
    image_dir_paths = []
    labels = config['model']['labels']

    #get image paths
    if os.path.isdir(image_dir_path): 
        for inp_file in os.listdir(image_dir_path):
            image_dir_paths += [os.path.join(image_dir_path, inp_file)]
    else:
        image_dir_paths += [image_dir_path]

    image_dir_paths = [inp_file for inp_file in image_dir_paths if\
            (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

    if savePredictionsAsXmlToo:
        #setup the directory to save at
        parent_dir = os.path.abspath(os.path.join(output_path, os.pardir))
        path_to_write_xml = os.path.join(parent_dir, "soft_bucket_labels")
        if not os.path.exists(path_to_write_xml):
            os.mkdir(path_to_write_xml)

    for image_dir_path in image_dir_paths:
        print(("Processing ", image_dir_path))

        #load and check the image, if not good continue with the other images
        image = cv2.imread(image_dir_path)
        if image is None:
            image = cv2.imread(image_dir_path)
            if image is None:
                print("Couldn't read image jumping to next!")
                continue

        boxes, filtered_boxes = detectionModel.predict(image, obj_threshold=obj_thresh,
                             nms_threshold=0.01, is_filter_bboxes=False,
                             shovel_type=shovel_type)
        boxes += filtered_boxes

        #visualize the predictions
        image = draw_boxes(image, boxes, labels,
                           score_threshold=obj_thresh) 

        #save the results
        path_to_save = os.path.join(output_path, image_dir_path.split('/')[-1])
        cv2.imwrite(path_to_save, np.uint8(image))         

        if savePredictionsAsXmlToo:
            gen_xml_file(image_dir_path, boxes, labels, path_to_write_xml,
                         excluded_classes=["Tooth", "Toothline"])


def predictOnH5Dir(h5s_dir_path, output_path, config, obj_threshold, detectionModel):
    h5_files = os.listdir(h5s_dir_path)

    for file_name in h5_files:
        filepath = os.path.join(h5s_dir_path, file_name)
        
        for i in range(config['train']['num_samples_in_h5']):
            detectionModel.predict_on_h5(
                filepath,
                i,
                output_path,
                sequence_length=config['model']['last_sequence_length'],
                stride=config['model']['stride'],
                obj_threshold=obj_threshold,
                nms_threshold=0.01)


def evaluateOnLabeledTestSet(config, model):
    valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'], 
                                                config['valid']['valid_image_folder'], 
                                                config['model']['labels'])

    model.batch_size = config['train']['batch_size']
    model.sequence_length = 1

    generator_config = {
        'IMAGE_H'         : model.input_size, 
        'IMAGE_W'         : model.input_size,
        'GRID_H'          : model.grid_h,  
        'GRID_W'          : model.grid_w,
        'BOX'             : model.nb_box,
        'LABELS'          : model.labels,
        'CLASS'           : len(model.labels),
        'ANCHORS'         : model.anchors,
        'BATCH_SIZE'      : model.batch_size,
        'TRUE_BOX_BUFFER' : model.max_box_per_image,
        'SEQUENCE_LENGTH' : model.sequence_length
    }    
    valid_generator = BatchGenerator(valid_imgs,
                                 generator_config, 
                                 norm=model.feature_extractor.normalize,
                                 jitter=False)   
    ave_precisions = model.evaluate(valid_generator, iou_threshold=0.3,
                                   score_threshold=0.2)

    print("ave precisions: ", ave_precisions)
    print('mAP: {:.4f}'.format(sum(ave_precisions.values()) / len(ave_precisions)))   


def _main_(args):
    with open(args.conf) as config_buffer:    
        config = json.load(config_buffer)
    pp.pprint(config)

    # keras.backend.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)))

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])
    yolo.load_weights(args.weights)

    ###############################
    #   Decide what to predict on
    ###############################
    if args.input[-4:] in ['.mp4', '.avi', '.mov']:
        predictOnVideo(args.input, args.output, config, maxFramesToRunOn=5000,
                       detectionModel=yolo)
    else:
        first_file_name = os.listdir(args.input)[0]

        # predict on folder of images and save images with overlayed bboxes
        if first_file_name[-4:] in [".jpg", ".png"]:
            predictOnImageDir(args.input, args.output, config,
                              savePredictionsAsXmlToo=args.save_soft, detectionModel=yolo)
        elif first_file_name[-4:] in [".h5", ".hdf5"]:
            predictOnH5Dir(args.input, args.output, config, obj_threshold=obj_thresh,
                           detectionModel=yolo)
        else:
            print('input -i argument extension did not match what we expected')

    if args.evaluate:  # Calculate mAp on a labeled set if evaluating quantitatively.
        evaluateOnLabeledTestSet(config, yolo)
      

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)

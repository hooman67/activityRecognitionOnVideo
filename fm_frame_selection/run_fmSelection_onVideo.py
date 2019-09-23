import os
import sys

import argparse
import json
import pprint as pp
from tqdm import tqdm

import cv2
import numpy as np

import tensorflow as tf
from keras import backend as K


sys.path.append('../')

from fm_frame_selection import FmRecord, HsFmFrameSelector
from yolo.frontend import YOLO
from rnn.rnn_train import TrackerModel
from utils.xml_utils import gen_xml_file
from utils.utils import draw_boxes


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

obj_thresh = 0.2  #0.05 final for hydraulics   #0.2
class_obj_threshold = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]  # [0.5, 0.3, 0.3, 0.3, 0.05, 0.3]  final for hydraulics

shovel_type = "Cable"
if shovel_type == "Cable": num_teeth = 8
elif shovel_type == "Hydraulic": num_teeth = 6
elif shovel_type == "Backhoe": num_teeth = 6


argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')
argparser.add_argument('-c', '--conf', required=True, help='path to configuration file')
argparser.add_argument('-w', '--weights', required=True, help='path to pretrained weights')
argparser.add_argument('-i', '--input', required=True,
        help='path to a video (mp4 format)')
argparser.add_argument('-o', '--output', type=str, required=True,
        help='path to save images/video.')
argparser.add_argument('-t', '--max_min_toothLength', required=True,
        help='maxAllowable_min_toothLength for FM frame selection')
argparser.add_argument('-b', '--buffer_size', required=True,
        help='decision making buffer size FM frame selection')
argparser.add_argument('-r', '--is_rnn', required=True,
        help='is this an rnn model or just yolo')



def runFMSelectionOnVideo(video_path, output_path, config, maxFramesToRunOn, decisionMakingBufferLength, maxAllowableMinToothLength, detectionModel):  
    skip_rate = 3
    frame_rate = 30


    video_reader = cv2.VideoCapture(video_path)

    #get max number of frames to predict on
    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))    
    frames_to_predict = min(maxFramesToRunOn, nb_frames-30) 


    hsFmFrameSelector = HsFmFrameSelector(decisionMakingBufferLength, maxAllowableMinToothLength, output_path, video_path)

    
    for i in tqdm(list(range(frames_to_predict))):
        _, image = video_reader.read()

        if i % skip_rate == 0:

            boxes, filtered_boxes = detectionModel.predict(image, obj_threshold=obj_thresh,
                                 nms_threshold=0.01, is_filter_bboxes=True,
                                 shovel_type=shovel_type,
                                 class_obj_threshold=class_obj_threshold)
          
            #we do this to draw both original and filtered boxes. Draw boxes knows to draw them with different thicknesses.
            #boxes += filtered_boxes
            image = draw_boxes(image, boxes + filtered_boxes,
                               config['model']['labels'],
                               score_threshold=obj_thresh,
                               class_obj_threshold=class_obj_threshold)


            hsFmFrameSelector.update( FmRecord(image, i, boxes, filtered_boxes) )

      

    hsFmFrameSelector.summerizeTheResults()

    video_reader.release()

    print("Ran prediction on this video:\n" + str(video_path) + '\n and saved the results inside\n' + str(output_path))


def _main_(args):
    with open(args.conf) as config_buffer:    
        config = json.load(config_buffer)

    pp.pprint(config)


    ###############################
    #   Load the model 
    ###############################
    if args.is_rnn: #this is a yolo+rnn model
        #TODO
        config['train']['batch_size'] = 1

        model = TrackerModel(config, is_inference=True)

        model.full_model.load_weights(args.weights)

        #reading the trained weights from training model and writing them to the inference model
        model.set_inference_weights()

        #reset the hidden state of lstm
        model.full_model_inference.reset_states()

        #this supposedly releases the memory of the trained model now that we no longer need it. 
        model.full_model = None


    else: #this is just yolo model
        model = YOLO(backend             = config['model']['backend'],
                    input_size          = config['model']['input_size'], 
                    labels              = config['model']['labels'], 
                    max_box_per_image   = config['model']['max_box_per_image'],
                    anchors             = config['model']['anchors'])

        model.load_weights(args.weights)



    ###############################
    #   Decide what to predict on
    ###############################
    if args.input[-4:] in ['.mp4', '.avi', '.mov']:

        #maxAllowable_min_toothLength = 40 #60 good for 1_20161115-110100_0001n0.avi  #40 good for 151700_0001n0
        runFMSelectionOnVideo(
            args.input,
            args.output,
            config,
            maxFramesToRunOn=5000,
            decisionMakingBufferLength= int(args.buffer_size),
            maxAllowableMinToothLength = int(args.max_min_toothLength),
            detectionModel=model
        )

    else:
        print('input -i argument extension did not match mp4, avi, or mov video formats')


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)

#! /usr/bin/env python


import os

import argparse
import json
import pprint as pp
import keras
from keras import backend as K
import tensorflow as tf

from freeze_session import freeze_session
from frontend import YOLO



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


argparser = argparse.ArgumentParser(
    description='Convert a trained model defined in Keras into Tensorflow')
argparser.add_argument('-c', '--conf', required=True, help='path to configuration file')
argparser.add_argument('-w', '--weights', required=True, help='path to pretrained weights')
argparser.add_argument('-o', '--output', type=str, required=True,
        help='path to save images/video.')



def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    output_path = args.output


    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)



    ###############################
    #   load the model 
    ###############################

    # keras.backend.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)))

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    yolo.load_weights(weights_path)


    yolo_inf = yolo.get_inference_model()
    yolo_inf.load_weights(weights_path)

    frozen_graph = freeze_session(K.get_session(),
                          output_names=[out.op.name for out in yolo_inf.outputs])
    
    tf.train.write_graph(frozen_graph, output_path, "convertedModel.pb", as_text=False)






if __name__ == '__main__':
    args = argparser.parse_args()
    pp.pprint(vars(args))
    _main_(args)

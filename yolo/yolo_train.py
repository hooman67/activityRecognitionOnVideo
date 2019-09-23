#! /usr/bin/env python

import argparse
import os
import sys
import json
import pprint as pp

import numpy as np
import tensorflow as tf
import mlflow
from mlflow import log_artifact

sys.path.append('../')
from utils.preprocessing import parse_annotation
from yolo.frontend import YOLO


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')
argparser.add_argument('-c', '--conf', help='path to configuration file')
argparser.add_argument('-d', '--description', type=str, required=True,
                       help='description of a current experiment to save for mlflow.')


def _main_(args):
    from keras.backend.tensorflow_backend import set_session
    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # sessconfig.gpu_options.per_process_gpu_memory_fraction=0.1
    # sessconfig.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=sessconfig)
    set_session(sess)

    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    pp.pprint(config)

    train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'],
                                                config['train']['train_image_folder'],
                                                config['model']['labels'])

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(str(config['valid']['valid_annot_folder'])):
        valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'],
                                                    config['valid']['valid_image_folder'],
                                                    config['model']['labels'])
    else:
        print("Splitting into train and validation")
        train_valid_split = int(0.95 * len(train_imgs))
        np.random.shuffle(train_imgs)

        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]

    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(
            set(train_labels.keys()))

        print(('Seen labels:\t', train_labels))
        print(('Given labels:\t', config['model']['labels']))
        print(('Overlap labels:\t', overlap_labels))

        if len(overlap_labels) < len(config['model']['labels']):
            print('Some labels have no annotations! Please revise the list of labels '
                  'in the config.json file!')
            return
    else:
        print('No labels are provided. Train on all seen labels.')
        config['model']['labels'] = list(train_labels.keys())

    yolo = YOLO(backend=config['model']['backend'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    if os.path.exists(str(config['train']['pretrained_weights'])):
        print(("Loading pre-trained weights in", config['train']['pretrained_weights']))
        yolo.load_weights(config['train']['pretrained_weights'])

    print("\nStarting training...")
    mlflow.start_run()
    #mlflow.set_experiment(args.description)
    config_filename = args.conf.split('/')[-1]
    log_artifact("../" + config_filename)

    yolo.train(train_imgs=train_imgs,
               valid_imgs=valid_imgs,
               train_times=config['train']['train_times'],
               valid_times=config['valid']['valid_times'],
               nb_epochs=config['train']['nb_epochs'],
               learning_rate=config['train']['learning_rate'],
               batch_size=config['train']['batch_size'],
               warmup_epochs=config['train']['warmup_epochs'],
               object_scale=config['train']['object_scale'],
               no_object_scale=config['train']['no_object_scale'],
               coord_scale=config['train']['coord_scale'],
               class_scale=config['train']['class_scale'],
               saved_weights_name=config['train']['saved_weights_name'],
               debug=config['train']['debug'],
               full_log_dir=config['train']['tensorboard_log_dir'],
               early_stop_patience=config['train']['early_stop_patience'],
               early_stop_min_delta=config['train']['early_stop_min_delta'],
               learning_rate_decay_factor=config['train']['learning_rate_decay_factor'],
               learning_rate_decay_patience=config['train'][
                   'learning_rate_decay_patience'],
               learning_rate_decay_min_lr=config['train']['learning_rate_decay_min_lr'])
    mlflow.end_run()


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)

import os, sys
import numpy as np
import h5py
import argparse
import json
import pprint as pp
from time import time

sys.path.append('../')
from utils.preprocessing import parse_annotation, parse_annotation_activity
from rnn.preprocessing_rnn import SequenceGeneratorForH5Activity,\
                                  SequenceGeneratorForH5Detection
from yolo.frontend import YOLO

argparser = argparse.ArgumentParser(
    description='Create H5 files with image and label sequences.'
                'Each h5 file will have size_of_h5 number of sequences,'
                'it will include output tensor that will be used to train LSTM.')
argparser.add_argument('-c', '--conf', required=True,
                       help='path to configuration file')
argparser.add_argument('-t', '--target_type', type=str, required=True,
                       choices=["detection", "activity"],
                       help="data for either object detection or activity recogn.")
argparser.add_argument('-y', '--is_yolo_feats', action='store_true',
                       help='whether to output yolo features or not')
argparser.add_argument('-w', '--weights',
                       help='path to pretrained YOLO weights')
argparser.add_argument('-z', '--is_test_data', action='store_true',
                       help='whether to generate test data (instead of train data)')


"""
In general there is a big trade-off btw compression ratio and compression time.
lzf is faster compression IO but slightly lower compression ratio wrt gzip.
Chunking should be optimized when using comporession. During read, the whole
chunk is read at once.
"""
#comp_kwargs = {'compression': 'gzip', 'compression_opts': 1} # default is 4
COMP_KWARGS = {'compression': 'lzf'}

def load_act_data_generators(generator_config):
    train_annots = parse_annotation_activity(generator_config['ANNOT_PATH'])
    activity_folder_val = os.path.join(generator_config['ANNOT_PATH'], "val")
    valid_annots = parse_annotation_activity(activity_folder_val)

    train_batch = SequenceGeneratorForH5Activity(train_annots, generator_config, norm=None,
                                         shuffle=True, augment=False)
    valid_batch = SequenceGeneratorForH5Activity(valid_annots, generator_config, norm=None,
                                         augment=False)
    return train_batch, valid_batch


def load_det_data_generators(generator_config, train_image_folder,
                             train_annot_folder):
    valid_image_folder = train_image_folder  # TODO: For now, for simplicity
    valid_annot_folder = train_annot_folder

    train_imgs, seen_train_labels = parse_annotation(train_annot_folder,
                                                     train_image_folder,
                                                     labels=generator_config[
                                                         'LABELS'])
    valid_imgs, seen_valid_labels = parse_annotation(valid_annot_folder,
                                                     valid_image_folder,
                                                     labels=generator_config[
                                                         'LABELS'])

    train_batch = SequenceGeneratorForH5Detection(train_imgs, generator_config, norm=None,
                                         shuffle=True, augment=False)
    valid_batch = SequenceGeneratorForH5Detection(valid_imgs, generator_config, norm=None,
                                         augment=False)
    return train_batch, valid_batch


def get_yolo_features(yolo, x_batch, generator_config):
    sequence_length = x_batch.shape[0]
    netouts = np.zeros((sequence_length, generator_config['GRID_H'],
                        generator_config['GRID_W'], generator_config['BOX'],
                        4 + 1 + generator_config['CLASS']),
                       dtype=np.float32)
    for i_seq in range(sequence_length):
        image = x_batch[i_seq, ...]
        image = image / 255.
        image = np.expand_dims(image, axis=0)

        dummy_array = np.zeros((1, 1, 1, 1, generator_config['TRUE_BOX_BUFFER'], 4))
        netout = yolo.model.predict([image, dummy_array])
        netouts[i_seq, ...] = netout

    return netouts


def create_act_seqs(args):
    config_path = args.conf
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    pp.pprint(config)
    output_path = config['train']['train_h5_folder']

    ANNOT_PATH = config['train']['train_annot_folder']
    IMAGE_H = config['model']['input_size']
    IMAGE_W = config['model']['input_size']
    LABELS = config['model']['labels']
    BATCH_SIZE = config['train']['batch_size']
    if BATCH_SIZE != 1:
        print("Changing batch size to 1!")
        BATCH_SIZE = 1
    SEQUENCE_LENGTH = config['model']['h5_sequence_length']
    CLASS = len(LABELS)

    generator_config = {
        'ANNOT_PATH': ANNOT_PATH,
        'IMAGE_H': IMAGE_H,
        'IMAGE_W': IMAGE_W,
        'LABELS': LABELS,
        'CLASS': len(LABELS),
        'BATCH_SIZE': BATCH_SIZE,
        'SEQUENCE_LENGTH': SEQUENCE_LENGTH + 5
    }

    if args.is_yolo_feats:
        GRID_H, GRID_W = int((IMAGE_H / 32)), int((IMAGE_W / 32))
        assert (IMAGE_H / 32. == GRID_H)
        NB_BOX = int((len(config['model']['anchors']) / 2))

    train_batch, valid_batch = load_act_data_generators(generator_config)
    print("Length of Generators: ", len(train_batch), len(valid_batch))
    if args.is_test_data:
        print("\nCreating test data!")
        train_batch = valid_batch

    if args.is_yolo_feats:
        yolo = YOLO(backend=config['model']['backend'],
                    input_size=config['model']['input_size'],
                    labels=config['model']['labels'],
                    max_box_per_image=config['model']['max_box_per_image'],
                    anchors=config['model']['anchors'])
        yolo.load_weights(args.weights)

    size_of_h5 = config['train']['num_samples_in_h5']
    num_of_h5s = int((len(train_batch) / size_of_h5))

    print("\nCreating %d h5 files...\n" % num_of_h5s)
    for i in range(num_of_h5s):
        start_time = time()
        x_batches = np.zeros((size_of_h5, SEQUENCE_LENGTH, IMAGE_H, IMAGE_W, 3),
                             dtype=np.uint8)
        if args.is_yolo_feats:
            yolo_out = np.zeros(
                (size_of_h5, SEQUENCE_LENGTH, GRID_H, GRID_W, NB_BOX, 4 + 1 + CLASS),
                dtype=np.float32)
        act_batches = np.zeros((size_of_h5, len(LABELS)), dtype=np.float32)
        print("Doing %d-th h5 file." % i)
        for j in range(size_of_h5):
            # the idx passed as argument isn't used, idx is kept inside state of SeqGenerator
            x_batch, act_batch = train_batch[i * size_of_h5 + j]
            if not x_batch.any():  # or not b_batch.any() or not y_batch.any():
                print("\nSomething is wrong! Images are all zeros.")
                continue

            x_batches[j, ...] = x_batch  # x_batches are the input images
            act_batches[j, ...] = act_batch

            if args.is_yolo_feats:
                # tensor output of running yolo
                netouts = get_yolo_features(yolo, x_batch[0, ...], generator_config)
                yolo_out[j, ...] = netouts

        filename = "sequences_%d.h5" % i
        h5path = os.path.join(output_path, filename)
        with h5py.File(h5path, 'w') as f:
            f.create_dataset('x_batches', data=x_batches, dtype='uint8',
                             chunks=(1, 1, IMAGE_H, IMAGE_W, 3), **COMP_KWARGS)
            if args.is_yolo_feats:
                f.create_dataset('yolo_out', data=yolo_out,
                                 dtype='float32') # , **comp_kwargs)
            f.create_dataset('act_batches', data=act_batches,
                             dtype='float32')  # , **comp_kwargs)
        print("One h5 file time: %.3f" % (time() - start_time))
    return


def create_det_seqs(args):
    config_path = args.conf
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    pp.pprint(config)
    output_path = config['train']['train_h5_folder']

    IMAGE_H = config['model']['input_size']
    IMAGE_W = config['model']['input_size']
    GRID_H, GRID_W = int((IMAGE_H / 32)), int((IMAGE_W / 32))
    assert (IMAGE_H / 32. == GRID_H)
    NB_BOX = int((len(config['model']['anchors']) / 2))
    LABELS = config['model']['labels']
    ANCHORS = config['model']['anchors']
    BATCH_SIZE = config['train']['batch_size']
    if BATCH_SIZE != 1:
        print("Changing batch size to 1!")
        BATCH_SIZE = 1
    SEQUENCE_LENGTH = config['model']['h5_sequence_length']
    CLASS = len(LABELS)
    MAX_BOX_PER_IMAGE = config['model']['max_box_per_image']

    generator_config = {
        'IMAGE_H': IMAGE_H,
        'IMAGE_W': IMAGE_W,
        'GRID_H': GRID_H,
        'GRID_W': GRID_W,
        'BOX': NB_BOX,
        'LABELS': LABELS,
        'CLASS': len(LABELS),
        'ANCHORS': ANCHORS,
        'BATCH_SIZE': BATCH_SIZE,
        'TRUE_BOX_BUFFER': MAX_BOX_PER_IMAGE,
        'SEQUENCE_LENGTH': SEQUENCE_LENGTH + 5
    }

    train_batch, valid_batch = load_det_data_generators(
        generator_config, config['train']['train_image_folder'],
        config['train']['train_annot_folder'])
    print("Length of Generators", len(train_batch), len(valid_batch))
    if args.is_test_data:
        print("\nCreating test data!")
        train_batch = valid_batch

    size_of_h5 = config['train']['num_samples_in_h5']

    if args.is_yolo_feats:
        yolo = YOLO(backend=config['model']['backend'],
                    input_size=config['model']['input_size'],
                    labels=config['model']['labels'],
                    max_box_per_image=config['model']['max_box_per_image'],
                    anchors=config['model']['anchors'])
        yolo.load_weights(args.weights)

    num_of_h5s = int((len(train_batch) / size_of_h5))
    print("\nCreating %d h5 files...\n" % num_of_h5s)
    for i in range(num_of_h5s):
        x_batches = np.zeros((size_of_h5, SEQUENCE_LENGTH, IMAGE_H, IMAGE_W, 3),
                             dtype=np.uint8)
        if args.is_yolo_feats:
            yolo_out = np.zeros(
                (size_of_h5, SEQUENCE_LENGTH, GRID_H, GRID_W, NB_BOX, 4 + 1 + CLASS),
                dtype=np.float32)
        b_batches = np.zeros((size_of_h5, 1, 1, 1, 1, MAX_BOX_PER_IMAGE, 4),
                             dtype=np.float32)
        y_batches = np.zeros((size_of_h5, GRID_H, GRID_W, NB_BOX, 4 + 1 + CLASS),
                             dtype=np.float32)
        print("Doing %d-th h5 file." % i)
        j = 0
        while j < size_of_h5:
            # the idx passed as argument isn't used, idx is kept inside state of SeqGenerator
            [x_batch, b_batch], y_batch = train_batch[i * size_of_h5 + j]
            if not x_batch.any():  # or not b_batch.any() or not y_batch.any():
                print("\nSomething is wrong. Images are all zeros.")
                continue

            x_batches[j, ...] = x_batch  # x_batches are the input images
            b_batches[j, ...] = b_batch

            # these are the GT labels encoded into YOLO outputMap format
            y_batches[j, ...] = y_batch
            j += 1

            if args.is_yolo_feats:
                # tensor output of runing yolo
                netouts = get_yolo_features(yolo, x_batch[0, ...], generator_config)
                yolo_out[j, ...] = netouts

        filename = "sequences_%d.h5" % i
        h5path = os.path.join(output_path, filename)
        with h5py.File(h5path, 'w') as f:
            f.create_dataset('x_batches', data=x_batches, dtype='uint8',  # , **comp_kwargs)
                            chunks=(1, 1, IMAGE_H, IMAGE_W, 3), **COMP_KWARGS)
            if args.is_yolo_feats:
                f.create_dataset('yolo_out', data=yolo_out,
                                 dtype='float32')  # , **comp_kwargs)
            f.create_dataset('b_batches', data=b_batches,
                             dtype='float32')  # , **comp_kwargs)
            f.create_dataset('y_batches', data=y_batches,
                             dtype='float32')  # , **comp_kwargs)
    return


if __name__ == '__main__':
    args = argparser.parse_args()
    if args.is_yolo_feats:
        raise NotImplementedError("Didn't finish full implementation of this feature")
    pp.pprint(vars(args))
    if args.target_type == "detection":
        create_det_seqs(args)
    elif args.target_type == "activity":
        create_act_seqs(args)

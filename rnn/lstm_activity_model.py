import argparse
import pprint as pp
import json

import cv2
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.layers import Conv2D, Dense, Flatten, Input, ConvLSTM2D
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.utils import multi_gpu_model

from lstm_base_model import LSTMBaseModel
from utils.callbacks import ValOnlyProgbarLogger, DecayLR, \
    MultiGPUCheckpoint, TrainValTensorBoard_HS
from preprocessing_rnn import H5ReaderGeneratorActivity
from yolo.frontend import YOLO

argparser = argparse.ArgumentParser(
    description='Train and validate LSTM Activity recognition model.')
argparser.add_argument('-c', '--conf', help='path to configuration file')


class LSTMActivityModel(LSTMBaseModel):
    """
    Creates a Convolutional LSTM on top of YOLO feature maps in order to
    predict probability distribution (softmax) over activities.
    """
    def __init__(self, config, yolo_detector, is_testing=False):
        super().__init__(config, yolo_detector)

        self.activity_labels = config['model']['activity_labels']
        self.nb_activity_class = len(config['model']['activity_labels'])
        # TODO: per class loss weights

        self.training_model = self.create_training_model()
        if is_testing:
            self.testing_model = self.create_testing_model()
        self.steps_per_epoch = -1

    def create_training_model(self):
        input_images = Input(batch_shape=
                             (self.batch_size, self.sequence_length, self.image_h,
                              self.image_w, 3),
                             name='images_input')

        feature_detector_model = self.detector.get_feature_model(
            is_before_activation=False)

        # due to the fact that we can't fit the whole model into GPU memory, we
        # only train the LSTM layers and not the YOLO layers
        feature_detector_model = self.freeze_layers(feature_detector_model)

        if self.is_print_model_summary:
            print("\nSummary of feature detector:")
            feature_detector_model.summary()

        # run the yolo bb on each image, and stack up the results to be fed to LSTM
        yolo_feats_seq = TimeDistributed(feature_detector_model,
                                         name='each_frame_feats') \
            (input_images)

        recurrent_state = ConvLSTM2D(128, (1, 1), strides=(1, 1), padding='same',
                                     return_sequences=True, name='conv_lstm_1')(
            yolo_feats_seq)
        recurrent_state = ConvLSTM2D(128, (1, 1), strides=(1, 1), padding='same',
                                     return_sequences=False, name='conv_lstm_2')(
            recurrent_state)
        conv_act = Conv2D(16, (1, 1), strides=(1, 1), padding='same',
                          name='activity_conv')(
            recurrent_state)
        flattened_conv_act = Flatten()(conv_act)
        intermediate_dense = Dense(self.nb_activity_class*16, activation='relu',
                                   name='dense_1') \
            (flattened_conv_act)
        output_act = Dense(self.nb_activity_class, activation='softmax',
                           name='dense_2') \
            (intermediate_dense)
        model = Model([input_images], [output_act],
                      name='activity_model')

        if self.is_print_model_summary:
            print("\nFull MODEL:")
            model.summary()

        return model

    def create_testing_model(self):
        """ In comparison to the training model which takes a fixed number of
        images as input (sequence_length), this model takes a single image which
        can be run online for inference. """
        K.set_learning_phase(0)
        input_images = Input(batch_shape=
                             (self.batch_size, 1, self.image_h, self.image_w, 3),
                             name='images_input')
        feature_detector_model = self.detector.get_feature_model(
            is_before_activation=False)

        yolo_feats_seq = TimeDistributed(feature_detector_model,
                                         name='each_frame_feats')(input_images)

        recurrent_state = ConvLSTM2D(128, (1, 1), strides=(1, 1), padding='same',
                                     return_sequences=True, stateful=True,
                                     name='conv_lstm_1')(yolo_feats_seq)
        recurrent_state = ConvLSTM2D(128, (1, 1), strides=(1, 1), padding='same',
                                     return_sequences=False, stateful=True,
                                     name='conv_lstm_2')(recurrent_state)

        conv_act = Conv2D(16, (1, 1), strides=(1, 1), padding='same',
                          name='activity_conv')(
            recurrent_state)
        flattened_conv_act = Flatten()(conv_act)
        intermediate_dense = Dense(self.nb_activity_class*16, activation='relu',
                                   name='dense_1') \
            (flattened_conv_act)
        output_act = Dense(self.nb_activity_class, activation='softmax',
                           name='dense_2') \
            (intermediate_dense)
        model = Model([input_images], [output_act],
                      name='cnn_rnn_model')

        if self.is_print_model_summary:
            model.summary()
        return model

    def train(self):
        train_batch, valid_batch = self.load_data_generators_seq(
            self.batch_size, H5ReaderGeneratorActivity, self.activity_labels)
        print("Length of generators: %d, %d" % (len(train_batch), len(valid_batch)))
        x, a = train_batch[0]
        print("Input shapes train: ", x.shape, a.shape)
        x, a = valid_batch[0]
        print("Input shapes val: ", x.shape, a.shape)

        checkpoint_multi_hs = MultiGPUCheckpoint(
            '{name}_{{epoch:02d}}_BbAndLstm_valLoss-{{val_loss:.4f}}.h5'.format(
                name=self.saved_weights_name),
            verbose=1,
            save_best_only=False, )

        decay_lr = DecayLR(12, 15, 0.2)

        optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        print("Compiling a model...")
        with tf.device("/cpu:0"):
            self.training_model.compile(loss=keras.losses.categorical_crossentropy,
                                    optimizer=optimizer,
                                    metrics=['acc'])
        print("Successfuly compiled the full model!")

        self.steps_per_epoch = len(train_batch) if self.steps_per_epoch == -1 else \
            self.steps_per_epoch
        valid_steps_per_epoch = len(valid_batch) if self.steps_per_epoch == -1 else \
            self.steps_per_epoch
        print("Using %d/%d for train/val steps_per_epoch of %d batch_size!" % \
              (self.steps_per_epoch, valid_steps_per_epoch, self.batch_size))

        parallel_model = multi_gpu_model(self.training_model, gpus=2)
        parallel_model.compile(loss=keras.losses.categorical_crossentropy,
                               optimizer=optimizer,
                               metrics=['acc'])

        history = parallel_model.fit_generator(
            generator=train_batch,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            verbose=2 if self.debug else 1,
            validation_data=valid_batch,
            validation_steps=valid_steps_per_epoch,
            callbacks=[
                #early_stop,
                checkpoint_multi_hs,
                TrainValTensorBoard_HS(self.tensorboard_log_dir,
                                       write_graph=False, write_images=True),
                ValOnlyProgbarLogger(verbose=1, count_mode='steps'),
                decay_lr],
            #reduce_lr_hs],  # reduce_lr
            workers=4,
            max_queue_size=10,
            use_multiprocessing=True,
            shuffle=True,
            initial_epoch=self.initial_epoch)

        self.training_model.save(self.saved_weights_name + "_fullModel_final.h5")
        return history

    def predict(self, image):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.image_h, self.image_w))
        image = np.divide(image, 255., dtype=np.float32)

        input_image = image[:, :, ::-1]
        input_image = np.expand_dims(input_image, 0)
        input_image = np.expand_dims(input_image, 0)

        netout = self.training_model_inference.predict(input_image)
        activity_index = np.argmax(netout)
        activity = self.activity_labels[activity_index]
        return activity, netout


def main(args, is_debug=False):
    config_path = args.conf
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    pp.pprint(config)

    from keras.backend.tensorflow_backend import set_session
    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # sessconfig.gpu_options.per_process_gpu_memory_fraction=0.1
    sess = tf.Session(config=sessconfig)
    set_session(sess)
    detector = YOLO(backend=config['model']['backend'],
                    input_size=config['model']['input_size'],
                    labels=config['model']['labels'],
                    max_box_per_image=config['model']['max_box_per_image'],
                    anchors=config['model']['anchors'])
    detector.load_weights(config['model']['detector_weights_path'])
    tracker_model = LSTMActivityModel(config, detector)
    if is_debug:
        tracker_model.steps_per_epoch = 20
    pp.pprint(vars(tracker_model))
    history = tracker_model.train()
    return True


if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)

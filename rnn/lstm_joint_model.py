
import numpy as np
import cv2
import keras.backend as K
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.layers import Reshape, Conv2D, Dense, Flatten, Input, ConvLSTM2D

from lstm_base_model import LSTMBaseModel
from utils.utils import decode_netout
from utils.post_process_detections import filter_all_objects


class LSTMJointModel(LSTMBaseModel):
    """ Joint model of Activity and Detection LSTMs that's used for per-frame
    prediction. """

    def __init__(self, config_det, config_act, yolo_detector):
        super().__init__(config_det, yolo_detector)

        self.anchors = config_det['model']['anchors']
        self.true_box_buffer = config_det['model']['max_box_per_image']
        self.labels = config_det['model']['labels']
        self.nb_class = len(config_det['model']['labels'])
        self.nb_box = len(config_det['model']['anchors']) // 2

        self.activity_labels = config_act['model']['activity_labels']
        self.nb_activity_class = len(config_act['model']['activity_labels'])

        self.testing_model = self.create_joint_model()

    def create_joint_model(self):
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

        # First is detection LSTM branch
        recurrent_state_det = ConvLSTM2D(256, (1, 1), strides=(1, 1), padding='same',
                                     return_sequences=True, stateful=True,
                                     name='lstm_det_1')(yolo_feats_seq)
        recurrent_state_det = ConvLSTM2D(256, (1, 1), strides=(1, 1), padding='same',
                                     return_sequences=False, stateful=True,
                                     name='lstm_det_2')(recurrent_state_det)
        output_conv = Conv2D(
            self.nb_box * (4 + 1 + self.nb_class), (1, 1),
            strides=(1, 1), padding='same', kernel_initializer='lecun_normal',
            name='track_conv')(recurrent_state_det)
        output_reshaped = Reshape(
            (self.grid_h, self.grid_w, self.nb_box,
             4 + 1 + self.nb_class))(output_conv)

        # Second is Activity LSTM branch
        recurrent_state_act = ConvLSTM2D(128, (1, 1), strides=(1, 1), padding='same',
                                     return_sequences=True, stateful=True,
                                     name='lstm_act_1')(yolo_feats_seq)
        recurrent_state_act = ConvLSTM2D(128, (1, 1), strides=(1, 1), padding='same',
                                     return_sequences=False, stateful=True,
                                     name='lstm_act_2')(recurrent_state_act)

        conv_act = Conv2D(16, (1, 1), strides=(1, 1), padding='same',
                          name='activity_conv')(
            recurrent_state_act)
        flattened_conv_act = Flatten()(conv_act)
        intermediate_dense = Dense(self.nb_activity_class*16, activation='relu',
                                   name='dense_1') \
            (flattened_conv_act)
        output_act = Dense(self.nb_activity_class, activation='softmax',
                           name='dense_2') \
            (intermediate_dense)


        model = Model(input_images, [output_reshaped, output_act], name='joint_model')

        if self.is_print_model_summary:
            print("\nFull MODEL:")
            model.summary()
        model.summary()
        return model

    def predict(self, image, obj_threshold=0.3, nms_threshold=0.01,
                is_filter_bboxes=False, shovel_type="Hydraulic",
                num_teeth=6, class_obj_threshold=None):
        if class_obj_threshold is None:
            class_obj_threshold = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.image_h, self.image_w))
        image = np.divide(image, 255., dtype=np.float32)

        input_image = image[:, :, ::-1]  # TODO: Check whether this is necessary
        input_image = np.expand_dims(input_image, 0)
        input_image = np.expand_dims(input_image, 0)

        det_netout, act_netout = self.testing_model.predict(input_image)
        activity_index = np.argmax(act_netout)
        activity = self.activity_labels[activity_index]

        boxes = decode_netout(det_netout[0, ...], self.anchors, self.nb_class,
                              obj_threshold=obj_threshold,
                              nms_threshold=nms_threshold,
                              class_obj_threshold=class_obj_threshold)
        filtered_boxes = []
        if is_filter_bboxes:
            boxes, filtered_boxes = filter_all_objects(boxes, shovel_type,
                                                       num_teeth,
                                                       image_size=image_h)

        return boxes, filtered_boxes, activity, act_netout

import os
import sys

import cv2
from keras.models import Model
from keras.layers import Reshape, Conv2D, Input, Lambda
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import multi_gpu_model
import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import mlflow

sys.path.append('../')
from utils.preprocessing import BatchGenerator
from yolo.backend import TinyYoloFeature, FullYoloFeature, MobileNetFeature, \
    SqueezeNetFeature, Inception3Feature, VGG16Feature, ResNet50Feature
from utils.callbacks import ValOnlyProgbarLogger, EvaluateCallback, DecayLR, \
    MultiGPUCheckpoint, TrainValTensorBoard_HS, MLflowCheckpoint
from utils.post_process_detections import filter_all_objects
from utils.utils import decode_netout, compute_overlap, compute_ap, draw_boxes


class YOLO(object):
    def __init__(
        self,
        backend,
        input_size,
        labels,
        max_box_per_image,
        anchors):

        self.input_size = input_size

        self.labels = list(labels)
        self.nb_class = len(self.labels)
        self.nb_box = len(anchors) // 2
        self.class_wt = np.ones(self.nb_class, dtype='float32')
        self.anchors = anchors

        self.max_box_per_image = max_box_per_image

        # make the feature extractor layers
        input_image = Input(shape=(self.input_size, self.input_size, 3))
        self.true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4))

        if backend == 'Inception3':
            self.feature_extractor = Inception3Feature(self.input_size)
        elif backend == 'SqueezeNet':
            self.feature_extractor = SqueezeNetFeature(self.input_size)
        elif backend == 'MobileNet':
            self.feature_extractor = MobileNetFeature(self.input_size)
        elif backend == 'Full Yolo':
            self.feature_extractor = FullYoloFeature(self.input_size)
        elif backend == 'Tiny Yolo':
            self.feature_extractor = TinyYoloFeature(self.input_size)
        elif backend == 'VGG16':
            self.feature_extractor = VGG16Feature(self.input_size)
        elif backend == 'ResNet50':
            self.feature_extractor = ResNet50Feature(self.input_size)
        else:
            raise Exception(
                'Architecture not supported! Only support Full Yolo, Tiny Yolo, MobileNet, SqueezeNet, VGG16, ResNet50, and Inception3 at the moment!')

        print("Output Shape of feature extractor: ",
              self.feature_extractor.get_output_shape())
        self.grid_h, self.grid_w = self.feature_extractor.get_output_shape()
        features = self.feature_extractor.extract(input_image)

        # make the object detection layer
        output = Conv2D(self.nb_box * (4 + 1 + self.nb_class),
                        (1, 1), strides=(1, 1),
                        padding='same',
                        name='DetectionLayer',
                        kernel_initializer='lecun_normal')(features)
        output = Reshape((self.grid_h, self.grid_w, self.nb_box,
                          4 + 1 + self.nb_class))(output)
        output = Lambda(lambda args: args[0])([output, self.true_boxes])

        self.model = Model([input_image, self.true_boxes], output)

        # initialize the weights of the detection layer
        layer = self.model.layers[-4]
        weights = layer.get_weights()

        new_kernel = np.random.normal(size=weights[0].shape) / (self.grid_h * self.grid_w)
        new_bias = np.random.normal(size=weights[1].shape) / (self.grid_h * self.grid_w)

        layer.set_weights([new_kernel, new_bias])

        # print a summary of the whole model
        self.model.summary()

    def custom_loss(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]),
                                        (1, self.grid_h, self.grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

        cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1),
                            [self.batch_size, 1, 1, self.nb_box, 1])

        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)

        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)

        """
        Adjust prediction
        """
        ### adjust x and y      
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors,
                                                            [1, 1, 1, self.nb_box, 2])

        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2]  # relative position to the containing cell

        ### adjust w and h
        true_box_wh = y_true[...,
                      2:4]  # number of cells accross, horizontally and vertically

        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale

        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (
                    1 - y_true[..., 4]) * self.no_object_scale

        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * self.object_scale

        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(self.class_wt,
                                                true_box_class) * self.class_scale

        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < self.coord_scale / 2.)
        seen = tf.assign_add(seen, 1.)

        true_box_xy, true_box_wh, coord_mask = tf.cond(
            tf.less(seen, self.warmup_batches + 1),
            lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                     true_box_wh + tf.ones_like(true_box_wh) * \
                     np.reshape(self.anchors, [1, 1, 1, self.nb_box, 2]) * \
                     no_boxes_mask,
                     tf.ones_like(coord_mask)],
            lambda: [true_box_xy,
                     true_box_wh,
                     coord_mask])

        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

        loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (
                    nb_coord_box + 1e-6) / 2.
        loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (
                    nb_coord_box + 1e-6) / 2.
        loss_conf = tf.reduce_sum(
            tf.square(true_box_conf - pred_box_conf) * conf_mask) / (
                                nb_conf_box + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class,
                                                                    logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

        loss = tf.cond(tf.less(seen, self.warmup_batches + 1),
                       lambda: loss_xy + loss_wh + loss_conf + loss_class + 10,
                       lambda: loss_xy + loss_wh + loss_conf + loss_class)

        if self.debug:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_pred_box = tf.reduce_sum(
                tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

            current_recall = nb_pred_box / (nb_true_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)

            loss = tf.Print(loss, [loss_xy], message='\nLoss XY \t', summarize=1000)
            loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
            loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
            loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [current_recall], message='Current Recall \t',
                            summarize=1000)
            # loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)

        return loss

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)

    def train(
        self,
        train_imgs,
        valid_imgs,
        train_times,
        # the number of time to repeat the training set, often used for small datasets
        valid_times,
        # the number of times to repeat the validation set, often used for small datasets
        nb_epochs,  # number of epoches
        learning_rate,  # the learning rate
        batch_size,  # the size of the batch
        warmup_epochs,
        # number of initial batches to let the model familiarize with the new dataset
        object_scale,
        no_object_scale,
        coord_scale,
        class_scale,
        full_log_dir,
        early_stop_patience,
        early_stop_min_delta,
        learning_rate_decay_factor,
        learning_rate_decay_patience,
        learning_rate_decay_min_lr,
        saved_weights_name='best_weights.h5',
        debug=False,
        sequence_length=10):

        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.object_scale = object_scale
        self.no_object_scale = no_object_scale
        self.coord_scale = coord_scale
        self.class_scale = class_scale

        self.debug = debug

        self.full_log_dir = full_log_dir
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.learning_rate_decay_patience = learning_rate_decay_patience
        self.learning_rate_decay_min_lr = learning_rate_decay_min_lr

        ############################################
        # Make train and validation generators
        ############################################

        generator_config = {
            'IMAGE_H': self.input_size,
            'IMAGE_W': self.input_size,
            'GRID_H': self.grid_h,
            'GRID_W': self.grid_w,
            'BOX': self.nb_box,
            'LABELS': self.labels,
            'CLASS': len(self.labels),
            'ANCHORS': self.anchors,
            'BATCH_SIZE': self.batch_size,
            'TRUE_BOX_BUFFER': self.max_box_per_image,
            'SEQUENCE_LENGTH': self.sequence_length
        }

        train_generator = BatchGenerator(train_imgs,
                                         generator_config,
                                         norm=self.feature_extractor.normalize,
                                         debug=self.debug)
        valid_generator = BatchGenerator(valid_imgs,
                                         generator_config,
                                         norm=self.feature_extractor.normalize,
                                         jitter=False)

        self.warmup_batches = warmup_epochs * (
                    train_times * len(train_generator) + valid_times * len(
                valid_generator)) / 4
        print("Using %d warmup batches" % self.warmup_batches)

        ############################################
        # Define your callbacks
        ############################################

        # HS HSS With a patience of 100 you finish in 200 epochs so I changed it to 400
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=self.early_stop_min_delta,
                                   patience=self.early_stop_patience,
                                   verbose=1)

        # This didnt work with multi gpu
        checkpoint = ModelCheckpoint(
            '{name}_{{epoch:02d}}.h5'.format(name=saved_weights_name),
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            mode='min',
            period=1)

        # define by Anuar because above didn't work with multi GPU
        checkpoint_multi = MultiGPUCheckpoint(
            '{name}_{{epoch:02d}}_multi.h5'.format(name=saved_weights_name),
            verbose=1,
            save_best_only=True,
            mode='min',
            period=1)

        # defined by hs for best val
        checkpoint_multi_hs = MultiGPUCheckpoint(
            '{name}_{{epoch:02d}}_hsBb_valLoss-{{val_loss:.2f}}.h5'.format(
                name=saved_weights_name),
            verbose=1,
            save_best_only=True, )

        # defined by HS
        # HS HSS originally i used monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6
        # reduce_lr_hs = ReduceLROnPlateau(monitor='val_loss',
        #                                  factor=self.learning_rate_decay_factor,
        #                                  patience=self.learning_rate_decay_patience,
        #                                  min_lr=self.learning_rate_decay_min_lr)

        # written by Anuar
        evaluate_callback_train = EvaluateCallback(train_generator, self.evaluate)
        evaluate_callback_val = EvaluateCallback(valid_generator, self.evaluate)
        # written by Anuar
        decay_lr = DecayLR(30, 40, 0.2)

        mlflow_callback = MLflowCheckpoint("yolo_all_shovels")

        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                         decay=0.0)

        ############################################
        # Compile the model
        ############################################
        with tf.device("/cpu:0"):
            self.model.compile(loss=self.custom_loss, optimizer=optimizer)

        ############################################
        # Start the training process
        ############################################  

        steps_per_epoch = len(train_generator) * train_times

        parallel_model = multi_gpu_model(self.model, gpus=2)
        parallel_model.compile(loss=self.custom_loss, optimizer=optimizer)
        parallel_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=warmup_epochs + nb_epochs,
            verbose=2 if debug else 1,
            validation_data=valid_generator,
            validation_steps=len(valid_generator) * valid_times,
            callbacks=[
                early_stop,
                checkpoint_multi_hs,
                TrainValTensorBoard_HS(self.full_log_dir,
                                       write_graph=False, write_images=True),
                ValOnlyProgbarLogger(verbose=1, count_mode='steps'),
                # reduce_lr_hs,
                decay_lr,
                evaluate_callback_val,
                mlflow_callback],

            workers=4,
            max_queue_size=10,
            use_multiprocessing=True)

        self.model.save(saved_weights_name + "_final.h5")

        ############################################
        # Compute mAP on the validation set
        ############################################
        average_precisions = self.evaluate(valid_generator, iou_threshold=0.5,
                                           score_threshold=0.5)
        for label, average_precision in list(average_precisions.items()):
            print(self.labels[label], '{:.4f}'.format(average_precision))
        print('mAP: {:.4f}'.format(
            sum(average_precisions.values()) / len(average_precisions)))

        average_precisions = self.evaluate(valid_generator, iou_threshold=0.3,
                                           score_threshold=0.3)
        for label, average_precision in list(average_precisions.items()):
            print(self.labels[label], '{:.4f}'.format(average_precision))
        print('mAP: {:.4f}'.format(
            sum(average_precisions.values()) / len(average_precisions)))

    def evaluate(self,
                 generator,
                 iou_threshold=0.3,
                 score_threshold=0.3,
                 max_detections=100,
                 save_path=None):
        """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet
        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
        """
        # gather all detections and annotations
        all_detections = [[None for i in range(generator.num_classes())] for j in
                          range(generator.size())]
        all_annotations = [[None for i in range(generator.num_classes())] for j in
                           range(generator.size())]

        for i in range(generator.size()):
            raw_image = generator.load_image(i)
            raw_height, raw_width, raw_channels = raw_image.shape

            pred_boxes, filtered_boxes = self.predict(raw_image,
                                                      obj_threshold=score_threshold,
                                                      nms_threshold=0.1,
                                                      is_filter_bboxes=False,
                                                      shovel_type="Cable",
                                                      class_obj_threshold=None)

            score = np.array([box.score for box in pred_boxes])
            pred_labels = np.array([box.label for box in pred_boxes])

            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin * raw_width, box.ymin * raw_height,
                                        box.xmax * raw_width, box.ymax * raw_height,
                                        box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])

                # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes = pred_boxes[score_sort]

            # copy detections to all_detections
            for label in range(generator.num_classes()):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]

            annotations = generator.load_annotation(i)

            # copy detections to all_annotations
            for label in range(generator.num_classes()):
                if annotations.any():
                    all_annotations[i][label] = annotations[annotations[:, 4] == label,
                                                :4].copy()
                else:
                    all_annotations[i][label] = np.array([], dtype=np.int)

        # compute mAP by comparing all detections and all annotations
        average_precisions = {}

        for label in range(generator.num_classes()):
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(generator.size()):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]
                num_annotations += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue

                    overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives,
                                                    np.finfo(np.float64).eps)

            # compute average precision
            average_precision = compute_ap(recall, precision)
            average_precisions[label] = average_precision

        return average_precisions

    def predict(
        self,
        image,
        obj_threshold,
        nms_threshold,
        is_filter_bboxes,
        shovel_type,
        class_obj_threshold):

        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.feature_extractor.normalize(image)

        input_image = image[:, :, ::-1]
        input_image = np.expand_dims(input_image, 0)
        dummy_array = np.zeros((1, 1, 1, 1, self.max_box_per_image, 4))

        netout = self.model.predict([input_image, dummy_array])[0]

        boxes = decode_netout(netout, self.anchors, self.nb_class,
                              obj_threshold=obj_threshold,
                              nms_threshold=nms_threshold,
                              class_obj_threshold=class_obj_threshold)

        filtered_boxes = []
        if is_filter_bboxes:
            boxes, filtered_boxes = filter_all_objects(boxes, shovel_type,
                                                       image_size=image_h)

        return boxes, filtered_boxes

    def get_inference_model(self):
        inference_model = Model(self.model.input,
                                self.model.get_layer("reshape_1").output)
        return inference_model

    def get_feature_model(self, is_before_activation=True):
        if is_before_activation:
            feature_model = Model(self.feature_extractor.feature_extractor.inputs,
                                  self.feature_extractor.feature_extractor. \
                                  get_layer("conv_22").output)
        else:
            feature_model = Model(self.feature_extractor.feature_extractor.inputs,
                                  self.feature_extractor.feature_extractor. \
                                  get_layer("leaky_re_lu_22").output)
        return feature_model

    def predict_on_h5(self, h5_path, idx, path_to_save, sequence_length, stride, obj_threshold, nms_threshold):
        f = h5py.File(h5_path, 'r')
        x_batches = f["x_batches"]
        b_batches = f["b_batches"]
        y_batches = f["y_batches"]

        id_in_h5 = idx % x_batches.shape[0]
        x_batch = x_batches[id_in_h5, ...]  # read from disk
        b_batch = b_batches[id_in_h5, ...]
        y_batch = y_batches[id_in_h5, ...]

        x_batch = x_batch[::-1, ...][:sequence_length:stride][::-1, ...]
        image_id = -1
        image = x_batch[image_id, ...].copy()
        boxes, filtered_boxes = self.predict(image, obj_threshold=obj_threshold,
                                             nms_threshold=nms_threshold,
                                             is_filter_bboxes=False,
                                             shovel_type="Cable",
                                             class_obj_threshold=None)
        boxes += filtered_boxes
        image = draw_boxes(image, boxes, self.labels, score_threshold=obj_threshold)

        h5_name = h5_path.split('/')[-1]
        filepath = os.path.join(path_to_save, "pred_" + h5_name + str(idx) + ".jpg")
        cv2.imwrite(filepath, image)

        f.close()

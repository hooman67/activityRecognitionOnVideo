import os
import sys
import argparse
import json
import pprint as pp
from time import time

from tqdm import tqdm
import cv2
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

sys.path.append('../')
sys.path.append('../keypoints/lib')
from rnn.lstm_joint_model import LSTMJointModel
from rnn.lstm_detection_model import LSTMDetectionModel
from utils.utils import draw_boxes
from utils.utils_keypoints import init_pose_estimator,\
    get_tf_keypoints_network_session_info, get_tooth_wm_patches, map_keypoints,\
    visualize_keypoints, draw_keypoints
from keypoints.lib.core.inference import predict_on_images, tf_predict_on_images
import keypoints.lib.models as models  # this is called in `eval()`
from utils.timestamp import process_xml_to_dict
from yolo.frontend import YOLO

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


"""
sample command:
python rnn_predict.py \
-c /media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/yolo_v2_models/RNN/256_1x1_lstm_raw_images_2_sec_augment_after_activation_lr_reduced/config_full_yolo_new_rnn_copy.json \
-w /media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/yolo_v2_models/RNN/256_1x1_lstm_raw_images_2_sec_augment_after_activation_lr_reduced/ConvLSTM2D-Tracker-32.h5 \
-i /media/sf_N_DRIVE/videos/1089_SH02_DGM_Chile_Bucyrus/2018020711/1_20180207-112100_0001n0.avi  \
-o /media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/YOLO_v2_videos/detection_and_keypoints/testing \
--keypoints \
-ktf \
-m /home/cheng/Desktop/repos/bucket-tracking/models/pose_resnet_18.pb 
-s 
 

python rnn_predict.py \
-c /media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/yolo_v2_models/RNN/256_1x1_lstm_raw_images_2_sec_augment_after_activation_lr_reduced/config_full_yolo_new_rnn_copy.json \
-w /media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/yolo_v2_models/RNN/256_1x1_lstm_raw_images_2_sec_augment_after_activation_lr_reduced/ConvLSTM2D-Tracker-32.h5 \
-i /media/sf_N_DRIVE/videos/1089_SH02_DGM_Chile_Bucyrus/2018020711/1_20180207-112100_0001n0.avi  \
-o /media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/YOLO_v2_videos/detection_and_keypoints/testing \
--keypoints \
-cfg /home/cheng/Desktop/repos/bucket-tracking/keypoints/experiments/mpii/resnet50/96x192_d256x3_adam_lr1e-3_landmarks_try8_resnet_18.yaml \
-m /home/cheng/Desktop/repos/bucket-tracking/keypoints/output/mpii/pose_resnet_18/96x192_d256x3_adam_lr1e-3_landmarks_try8_resnet_18/final_state.pth.tar
-s 


/media/sf_N_DRIVE/videos/1084_Ahafo_Hydraulic_liebherr_R9400/2018012214/1_20180122-151700_0001n0.avi
/media/sf_N_DRIVE/videos/1016_CHM/1016_CHM_023.avi
/media/sf_N_DRIVE/videos/1004_ESP/S04/1004_ESP_S04_005.avi

/media/sf_N_DRIVE/videos/1087_Sh01_DGM_Chile_Bucyrus/2017110520/1_20171105-201900_0001n0.avi
uneven teeth lengths: /media/sf_N_DRIVE/videos/1089_SH02_DGM_Chile_Bucyrus/2018020711/1_20180207-112100_0001n0.avi

/media/sf_N_DRIVE/videos/1056_MEL_P\&H_4100XPC/SHE65_20160211/1_20160211-035900_0001n0.avi 

"""

argparser = argparse.ArgumentParser(
    description='Train and validate rnn model on any dataset')
argparser.add_argument('--conf_det',
                       help='path to object detection configuration file')
argparser.add_argument('--conf_act',
                       help='path to object detection configuration file')
argparser.add_argument('--weights_yolo',
                       help='path to pretrained single-frame yolo weights')
argparser.add_argument('--weights_det',
                       help='path to pretrained Detection LSTM weights')
argparser.add_argument('--weights_act',
                       help='path to pretrained Activity LSTM weights')
argparser.add_argument('-i', '--input', type=str, required=True,
                       help='path to h5 files or a video')
argparser.add_argument('-o', '--output', type=str, required=True,
                       help='path to save images/video.')
argparser.add_argument('-s', '--shovel_type', type=str, required=True,
                       choices=["Hydraulic", "Cable"],
                       help='Shovel type, used for post-processing')
argparser.add_argument('-k', '--keypoints', action='store_true',
                       help='activates keypoint prediction:'
                            'tooth tip and 4 WM landmarks')
argparser.add_argument('-ktf', '--keypoints-tf', action='store_true',
                       help='if args.keypoints=True, then this would'
                            'activate Tensorflow model instead of Pytorch')
argparser.add_argument('-cfg', '--keypoints_config', type=str,
                       help='Path to Pytorch keypoints config file')
argparser.add_argument('-m', '--keypoints_model', type=str,
                       help='Path to Pytorch/TF model weights')
argparser.add_argument('-f', '--frame', type=int, default=0,
                       help='Start at specific frame in a video')


def set_keras_session(gpu_mem_fraction=0.3):
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = False
    config_proto.gpu_options.per_process_gpu_memory_fraction = gpu_mem_fraction
    config_proto.intra_op_parallelism_threads = 2
    config_proto.inter_op_parallelism_threads = 2
    keras.backend.set_session(K.tf.Session(config=config_proto))


def find_act_in_frame(activity_dict, frame_num):
    gt_activity = "Swing E"  # if not found in ranges then it has to be Background
    all_activities = activity_dict['activities']
    for activity, ranges in all_activities.items():
        for t_range in ranges:
            if frame_num >= t_range[0] and frame_num <= t_range[1]:
                gt_activity = activity

    return gt_activity


def get_gt_activity(frame_num):
    gt_activity = None
    all_labels_folder = "/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/" \
                        "Anuar/data/optical/samans_activity"
    issue_name = args.input.split('/')[-1][:-4] + ".xml"
    issue_path = os.path.join(all_labels_folder, issue_name)
    if os.path.exists(issue_path):
        activity_dict = process_xml_to_dict(issue_path)
        gt_activity = find_act_in_frame(activity_dict, frame_num)

    if gt_activity == "Swing": gt_activity = "Swing F"
    return gt_activity


def draw_act_label(image, frame_num, activity_pred):
    colors = [(0, 255, 0), (0, 0, 255)]
    thickness = 3
    gt_activity = get_gt_activity(frame_num)
    if activity_pred == "Background": activity_pred = "Swing E"
    if gt_activity:
        cv2.putText(image, gt_activity, (image.shape[1]-160, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[0], thickness)
    cv2.putText(image, activity_pred, (image.shape[1]-160, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[1], thickness)

    return image, gt_activity


def get_pred_bar_graph(preds, act_labels, name):
    plt.tight_layout()
    fig = plt.figure()
    ax = fig.gca()
    matplotlib.rcParams.update({'font.size': 20})

    pred_ind = np.argmax(preds)
    colors = ['blue'] * len(act_labels)
    colors[pred_ind] = 'red'
    act_labels[0] = "Swing F"
    act_labels[2] = "Swing E"
    reorder = [4, 2, 1, 0, 3]  # for intuitive cycle ordering
    act_labels = np.array(act_labels)
    act_labels = act_labels[reorder]
    preds = preds[reorder]
    plt.barh(act_labels, preds, color=colors)
    plt.xlim([0., 1.])
    ax.axes.get_xaxis().set_visible(False)
    graph_path = "/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/data/optical/teeth_activity/hydraulic_full/output_folder/temp/%s.jpg" % name
    # dpi controls size of saved image, u can adjust it
    plt.savefig(graph_path, dpi=30, bbox_inches='tight')
    plt.close()

    return graph_path


def prepare_image_to_write(last_act, pred_buffer,
                           image_buffer, barplot_filepaths, i, j):
    """ Draws barplot and activity state (GT and best pred) onto image. """
    _, netout = pred_buffer[j]
    image, gt_activity = draw_act_label(image_buffer[j], i, last_act)
    barplot_path = barplot_filepaths[j]
    barplot = cv2.imread(barplot_path)
    y_margin = 70
    image[y_margin:barplot.shape[0]+y_margin, -barplot.shape[1]:, :] = barplot
    return image


def save_pred_time_graph(activity_labels, all_netouts, gt_activities,
                         all_last_acts, fig_path, reset_time):
    all_probs_seq = np.array(all_netouts)
    x = range(all_probs_seq[:, 0].shape[0])
    gt_activities_arr = np.zeros((len(x), len(activity_labels)))
    for i, act_ind in enumerate(gt_activities):
        gt_activities_arr[i, act_ind] = 1.
    last_acts_arr = np.zeros((len(x), len(activity_labels)))
    for i, act_ind in enumerate(all_last_acts):
        last_acts_arr[i, act_ind] = 0.99
    fig, ax = plt.subplots(len(activity_labels) + 1,
                           figsize=(36, 14), sharex=True,
                           gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 2]})
    for i in range(len(activity_labels)):
        ax[i].plot(x, all_probs_seq[:, i], 'b', label='prediction')
        if gt_activities_arr.any():
            ax[i].plot(x, gt_activities_arr[:, i], 'r-', label='GT')
        ax[i].plot(x, last_acts_arr[:, i], 'g.', label='Last')
        ax[i].set_xticks(np.arange(min(x), max(x)+1, 30.0))
        ax[i].set_xticklabels([i for i in x])
        ax[i].set_ylim(0., 1.05)
        ax[i].set_ylabel(activity_labels[i])
        ax[i].legend(loc="upper right")

    reorder = np.array([4, 3, 2, 1, 0])
    activity_labels = np.array(activity_labels)[reorder]
    all_probs_seq = all_probs_seq[:, reorder]
    for i, gt_act in enumerate(gt_activities):
        swap_gt = reorder[gt_act]
        gt_activities[i] = swap_gt

    window_size = reset_time // 3  # frames
    windows = range(all_probs_seq.shape[0])[::window_size]
    median_preds = all_probs_seq[:, 0].copy()
    last_preds = all_probs_seq[:, 0].copy()
    for i, j in zip(windows[:-1], windows[1:]):
        seq_window = all_probs_seq[i:j]
        max_inds = np.argmax(seq_window, axis=1)
        median_pred = np.median(max_inds)
        if median_pred % 0.5 == 0.:
            median_pred = max_inds[-1]
        median_pred = median_pred.astype(np.int)
        median_preds[i:j] = median_pred - 0.1
        last_pred = max_inds[-1]
        last_preds[i:j] = last_pred + 0.1

    max_inds = np.argmax(all_probs_seq, axis=1)
    ax[-1].plot(x, max_inds, 'bo')
    ax[-1].plot(median_preds, 'y', label='sliding median')
    ax[-1].plot(last_preds, 'g', label='last')
    ax[-1].plot(gt_activities, 'r', label='GT')
    ax[-1].set_ylabel("Max pred")
    ax[-1].set_ylim(-0.5, 5.)
    ax[-1].set_yticks(range(len(activity_labels)))
    ax[-1].set_yticklabels(activity_labels)
    ax[-1].legend(loc='upper right')
    ax[-1].grid(axis='y')
    fig.gca().xaxis.grid(True)
    fig.gca().yaxis.grid(True)
    fig.savefig(fig_path)
    plt.close(fig)
    return


def predictOnVideo(video_path, output_path, config, maxFramesToRunOn, detectionModel, keypoints_model, pose_config):
    video_out_name = config['model']['backend'] + "_rnnModel_" + \
                     video_path.split('/')[-1][:-4] + ".avi"
    video_out = os.path.join(output_path, video_out_name)
    video_reader = cv2.VideoCapture(video_path)
    if not video_reader.isOpened():
        raise Exception("Could not open the video %s" % video_path)
    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    ret = video_reader.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    if not ret:
        raise Exception("ERROR! Couldn't go to specific frame...")

    frames_to_predict = min(maxFramesToRunOn, nb_frames - 30)
    frame_rate = 30
    skip_rate = 3

    video_writer = cv2.VideoWriter(video_out,
                                   cv2.VideoWriter_fourcc(*'MPEG'),
                                   frame_rate / skip_rate,
                                   (frame_w, frame_h))

    detectionModel.activity_labels[0] = "Swing F"
    detectionModel.activity_labels[2] = "Swing E"
    activity_labels = detectionModel.activity_labels
    """ 
    Load the inference model
    the inference model works with a stream (undefined sequence length) 
    we read the trained weights into a training model, then copy them into 
    our inference model. """
    # reading the trained weights from training model and writing them to the inference model
    # detectionModel.set_inference_weights()

    # reset the hidden state of lstm
    detectionModel.testing_model.reset_states()

    # this supposedly releases the memory of the trained model now that we no longer need it.
    # detectionModel.full_model = None

    if args.keypoints:
        if args.keypoints_tf:
            patch_w, patch_h = 96, 192
            tf_session, x, y = get_tf_keypoints_network_session_info(keypoints_model)

    minute_timer = time()
    detection_minute_time, keypoints_minute_time, reset_minute_time = 0, 0, 0
    state_reset_timeout = config['model']['last_sequence_length']
    gt_activities = []
    all_netouts = []
    barplot_filepaths = []
    all_last_acts = []
    for i in tqdm(list(range(frames_to_predict))):
        _, image = video_reader.read()
        if i % 1800 == 0 and i != 0:  # one minute timer
            minute_end_time = time() - minute_timer
            minute_timer = time()
            print("\nOne minute took %.3f s" % minute_end_time)
            print("Detection: %.3f | Keypoints: %.3f s | Reset: %.3f" %\
                  (detection_minute_time, keypoints_minute_time, reset_minute_time))
            detection_minute_time, keypoints_minute_time, reset_minute_time = 0, 0, 0

        if i % state_reset_timeout == 0:
            reset_start_time = time()
            for layer in detectionModel.testing_model.layers:
                if layer.name.startswith("lstm_act"):
                    layer.reset_states()
            # detectionModel.testing_model.reset_states()
            reset_minute_time += time() - reset_start_time
            image_buffer, pred_buffer = [], []
            barplot_filepaths = []

        if i % skip_rate == 0:
            detection_start_time = time()
            boxes, filtered_boxes, activity, act_netout = detectionModel.predict(
                image, obj_threshold=obj_threshold, is_filter_bboxes=is_filter_bboxes,
                shovel_type=args.shovel_type, num_teeth=num_teeth,
                class_obj_threshold=class_obj_threshold
            )

            detection_minute_time += time() - detection_start_time

            keypoints_start_time = time()
            if args.keypoints:
                if args.keypoints_tf:
                    patches, patch_coords, coord_scales, tooth_tips_from_det =\
                        get_tooth_wm_patches(image, boxes, patch_h, patch_w, num_teeth)
                    keypoints = tf_predict_on_images(patches, tf_session, x, y, patch_w, patch_h)
                else:  # pytorch version
                    patch_w, patch_h = pose_config.MODEL.IMAGE_SIZE
                    patches, patch_coords, coord_scales, tooth_tips_from_det =\
                        get_tooth_wm_patches(image, boxes, patch_h, patch_w, num_teeth)
                    keypoints_model, pose_config = init_pose_estimator(args.keypoints_config,
                                                                       args.keypoints_model)
                    keypoints = predict_on_images(patches, keypoints_model)

                #visualize_keypoints(patches, keypoints)

                mapped_keypoints = map_keypoints(keypoints, patch_coords, coord_scales)
                image = draw_keypoints(image, mapped_keypoints, circle_radius=3)


            keypoints_minute_time += time() - keypoints_start_time

            # we do this to draw both original and filtered boxes. Draw boxes knows
            # to draw them with different thicknesses.
            boxes += filtered_boxes
            image = draw_boxes(
                image,
                boxes,
                config['model']['labels'],
                score_threshold=obj_threshold
            )

            barplot_path = get_pred_bar_graph(act_netout[0, :],
                                       activity_labels.copy(), i)
            barplot_filepaths.append(barplot_path)
            image_buffer.append(image)
            pred_buffer.append((activity, act_netout))
            all_netouts.append(act_netout[0, :])
            gt_activity = get_gt_activity(i)
            if gt_activity:
                gt_activity_ind = activity_labels.index(gt_activity)
                gt_activities.append(gt_activity_ind)

        if len(image_buffer) == int(state_reset_timeout / skip_rate) and i % skip_rate == 0:
            last_act = pred_buffer[-1][0]
            for j in range(int(state_reset_timeout / skip_rate)):
                all_last_acts.append(activity_labels.index(last_act))
                image = prepare_image_to_write(last_act,
                                               pred_buffer,
                                               image_buffer, barplot_filepaths,
                                               i, j)
                video_writer.write(np.uint8(image))

    video_reader.release()
    video_writer.release()

    fig_path = video_out[:-4] + "_" + str(frames_to_predict) + ".jpg"
    save_pred_time_graph(activity_labels, all_netouts, gt_activities,
                         all_last_acts, fig_path, state_reset_timeout)


def predictOnH5Dir(h5s_dir_path, output_path, config, detectionModel):
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


def _main_(args):
    with open(args.conf_det) as config_buffer:
        config_det = json.loads(config_buffer.read())
    with open(args.conf_act) as config_buffer:
        config_act = json.loads(config_buffer.read())
    pp.pprint(config_det)
    pp.pprint(config_act)

    config_det['train']['batch_size'] = 1
    set_keras_session(gpu_mem_fraction=0.35)

    # =================================================================
    # THIS MODEL WEIGHTS LOADING HAPPENS FROM 3 MODELS. TODO: Refactor this
    yolo_detector = YOLO(backend=config_det['model']['backend'],
                    input_size=config_det['model']['input_size'],
                    labels=config_det['model']['labels'],
                    max_box_per_image=config_det['model']['max_box_per_image'],
                    anchors=config_det['model']['anchors'])
    yolo_detector.load_weights(config_det['model']['detector_weights_path'])
    det_model = LSTMDetectionModel(config_det, yolo_detector)
    det_model.training_model.load_weights(args.weights_det)
    for layer in det_model.training_model.layers:
        if layer.name == "conv_lstm_1":
            layer.name = "lstm_det_1"
        if layer.name == "conv_lstm_2":
            layer.name = "lstm_det_2"

    model = LSTMJointModel(config_det, config_act, yolo_detector)
    print("\nDETECTION: ")
    for layer1 in det_model.training_model.layers:
       for layer2 in model.testing_model.layers:
           if layer1.name == layer2.name:
               print("Setting detection model layer: ", layer2.name)
               layer2.set_weights(layer1.get_weights())
    act_model = keras.models.load_model(args.weights_act)
    for layer in act_model.layers:
        if layer.name == "conv_lstm_1": layer.name = "lstm_act_1"
        if layer.name == "conv_lstm_2": layer.name = "lstm_act_2"
    for layer1 in act_model.layers:
        for layer2 in model.testing_model.layers:
            if layer1.name == layer2.name:
                print("Setting activity model layer: ", layer2.name)
                layer2.set_weights(layer1.get_weights())
    # ===============================================================

    if args.input[-4:] in ['.mp4', '.avi', '.mov']:
        predictOnVideo(args.input, args.output, config_act, 9000, model,
                       args.keypoints_model, args.keypoints_config)
    elif os.listdir(args.input)[0][-3:] in [".h5", ".hdf5"]:
        raise NotImplementedError("Not supported yet")
        predictOnH5Dir(args.input, args.output, config, detectionModel=tracker_model)
    else:
        raise Exception('input -i argument extension did not match what we expected')


if __name__ == '__main__':
    args = argparser.parse_args()
    obj_threshold = 0.2
    is_filter_bboxes = True
    if args.shovel_type == "Cable":
        num_teeth = 9  # 8 OR 9!
        class_obj_threshold = [0.4, 0.5, 0.4, 0.4, 0.4]
    elif args.shovel_type == "Hydraulic":
        num_teeth = 6
        class_obj_threshold = [0.5, 0.5, 0.4, 0.4, 0.4]
    elif args.shovel_type == "Backhoe":
        num_teeth = 6
        class_obj_threshold = [0.5, 0.5, 0.4, 0.4, 0.4]

    _main_(args)

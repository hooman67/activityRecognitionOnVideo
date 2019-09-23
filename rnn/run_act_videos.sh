#!/usr/bin/env bash

declare -a arrTrain=("/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111510/1_20161115-110100_0001n0.avi"
    "/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111612/1_20161116-125500_0001n0.avi"
    "/media/sf_N_DRIVE/videos/1071_PhuKham_Hydraulic_Komatsu_PC3000/2017011804/1_20170118-040301_0001n0.avi"
	"/media/sf_N_DRIVE/videos/1084_Ahafo_Hydraulic_liebherr_R9400/2018012210/1_20180122-101700_0001n0.avi"
	"/media/sf_N_DRIVE/videos/1016_CHM/avi/KAJF0576_20110311091324_avi.avi"
	"/media/sf_N_DRIVE/videos/1004_ESP/S05/1004_ESP_S05_002.avi"
)

declare -a arrAllTrain=("/media/sf_N_DRIVE/videos/1004_ESP/S05/1004_ESP_S05_002.avi"
#"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111506/1_20161115-073100_0001n0.avi"
#"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111508/1_20161115-081601_0001n0.avi"
#"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111508/1_20161115-084601_0001n0.avi"
#"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111510/1_20161115-110100_0001n0.avi"
#"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111512/1_20161115-121600_0001n0.avi"
#"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111512/1_20161115-123100_0001n0.avi"
#"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111512/1_20161115-131600_0001n0.avi"
"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111516/1_20161115-161600_0001n0.avi"
"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111516/1_20161115-171600_0001n0.avi"
"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111606/1_20161116-065500_0001n0.avi"
"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111606/1_20161116-071000_0001n0.avi"
"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111606/1_20161116-074000_0001n0.avi"
"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111606/1_20161116-075500_0001n0.avi"
"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111608/1_20161116-082500_0001n0.avi"
"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111608/1_20161116-094000_0001n0.avi"
"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111608/1_20161116-095500_0001n0.avi"
"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111610/1_20161116-101000_0001n0.avi"
"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111610/1_20161116-104000_0001n0.avi"
"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111610/1_20161116-105500_0001n0.avi"
"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111612/1_20161116-125500_0001n0.avi"
"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111612/1_20161116-131000_0001n0.avi"
"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111612/1_20161116-135500_0001n0.avi"
"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111614/1_20161116-141000_0001n0.avi"
"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111614/1_20161116-152500_0001n0.avi"
"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111614/1_20161116-154000_0001n0.avi"
"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111614/1_20161116-155500_0001n0.avi"
"/media/sf_N_DRIVE/videos/1071_PhuKham_Hydraulic_Komatsu_PC3000/2017011802/1_20170118-024801_0001n0.avi"
"/media/sf_N_DRIVE/videos/1071_PhuKham_Hydraulic_Komatsu_PC3000/2017011802/1_20170118-034801_0001n0.avi"
"/media/sf_N_DRIVE/videos/1071_PhuKham_Hydraulic_Komatsu_PC3000/2017011804/1_20170118-040301_0001n0.avi"
"/media/sf_N_DRIVE/videos/1071_PhuKham_Hydraulic_Komatsu_PC3000/2017011804/1_20170118-043301_0001n0.avi"
"/media/sf_N_DRIVE/videos/1071_PhuKham_Hydraulic_Komatsu_PC3000/2017011806/1_20170118-071100_0001n0.avi"
"/media/sf_N_DRIVE/videos/1071_PhuKham_Hydraulic_Komatsu_PC3000/2017011814/1_20170118-151400_0001n0.avi"
"/media/sf_N_DRIVE/videos/1071_PhuKham_Hydraulic_Komatsu_PC3000/2017011818/1_20170118-181400_0001n0.avi"
"/media/sf_N_DRIVE/videos/1071_PhuKham_Hydraulic_Komatsu_PC3000/2017011818/1_20170118-184400_0001n0.avi"
"/media/sf_N_DRIVE/videos/1071_PhuKham_Hydraulic_Komatsu_PC3000/2017011818/1_20170118-185900_0001n0.avi"
"/media/sf_N_DRIVE/videos/1071_PhuKham_Hydraulic_Komatsu_PC3000/2017011820/1_20170118-215900_0001n0.avi"
"/media/sf_N_DRIVE/videos/1071_PhuKham_Hydraulic_Komatsu_PC3000/2017011822/1_20170118-235500_0001n0.avi"
"/media/sf_N_DRIVE/videos/1084_Ahafo_Hydraulic_liebherr_R9400/2018012210/1_20180122-101700_0001n0.avi"
"/media/sf_N_DRIVE/videos/1084_Ahafo_Hydraulic_liebherr_R9400/2018012220/1_20180122-204600_0001n0.avi"
"/media/sf_N_DRIVE/videos/1084_Ahafo_Hydraulic_liebherr_R9400/2018012220/1_20180122-214600_0001n0.avi"
"/media/sf_N_DRIVE/videos/1016_CHM/avi/KAJF0576_20110311073340_avi.avi"
"/media/sf_N_DRIVE/videos/1016_CHM/avi/KAJF0576_20110311091324_avi.avi"
"/media/sf_N_DRIVE/videos/1016_CHM/avi/KAJF0576_20110311095601_avi.avi"
)

declare -a arrTest=(
	"/media/sf_N_DRIVE/videos/1084_Ahafo_Hydraulic_liebherr_R9400/2018012214/1_20180122-151700_0001n0.avi"
	"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111610/1_20161116-112500_0001n0.avi"
	"/media/sf_N_DRIVE/videos/1072_PintoValley_Hydraulic_Hitachi_EX5600/2016111508/1_20161115-090101_0001n0.avi"
	"/media/sf_N_DRIVE/videos/1016_CHM/1016_CHM_016.avi"
	"/media/sf_N_DRIVE/videos/1004_ESP/S04/1004_ESP_S04_017.avi")

#config="/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/data/optical/teeth_activity/hydraulic_full/output_folder/models_sixth_with_good_still_state/try6__2lstm-256-1b1-30frames.json"
#weightspath="/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/data/optical/teeth_activity/hydraulic_full/output_folder/models_sixth_with_good_still_state/lstm_act__05_BbAndLstm_valLoss-0.0087.h5"

weightsyolo="/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/hoomansBestHydraulicBBSoFar/full_yolo_bb_final.h5"
configdet="/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/forJointModel/Detection_LSTM_config.json"
configact="/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/forJointModel/Activity_LSTM_config.json"
weightsdet="/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/forJointModel/Detection_LSTM_valLoss-0.01.h5"
weightsact="/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/forJointModel/Activity_LSTM_valLoss-0.0087.h5"

keypointsmodel="/media/sf_N_DRIVE/randd/MachineLearning/Projects/ShovelMetrics/bucketTracking_1.0/keypointDetection_ResNet/try8_models/hydraulic_pose_resnet_18.pb"

outputfoldertest="/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/data/optical/teeth_activity/hydraulic_full/output_folder/models_sixth_with_good_still_state/reset_w_graph/test_videos"
outputfoldertrain="/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/forWACV/intro_images/keypoints_only"


for i in "${arrTrain[@]}"
do
    echo "$i"
	# python rnn_predict_joint.py -i $i -c $config -w $weightspath -o $outputfoldertrain -s Hydraulic
	python rnn_predict_joint.py -i $i -o $outputfoldertrain --weights_yolo $weightsyolo --conf_det $configdet --weights_det $weightsdet --conf_act $configact --weights_act $weightsact -k -ktf -m $keypointsmodel -s Hydraulic
done

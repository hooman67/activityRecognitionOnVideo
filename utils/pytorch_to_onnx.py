import sys
sys.path.append("../")

import torch
from torch.autograd import Variable

import keypoints.lib.models.pose_resnet as pr
from keypoints.lib.core.config import config, update_config


cfg = "/media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/bucket-tracking/keypoints/configs/hs.yaml"
model_file = "/media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/poseNet/hydraulic/try4(byAnuar)-OndataTry2_withoutAug_sigmaForHeatmaps1InsteadOf2/final_state.pth.tar"

image_w, image_h = 96, 192
onnx_model_output_path = "/media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/poseNet/hydraulic/try4(byAnuar)-OndataTry2_withoutAug_sigmaForHeatmaps1InsteadOf2/final_state.onnx"

update_config(cfg)
model = pr.get_pose_net(config, is_train=False)
model.load_state_dict(torch.load(model_file))

# will be the input to the model, (batch_size, n_channels, image_w, image_h)
dummy_input = Variable(torch.randn(1, 3, image_h, image_w))
input_names = ["input_image"]
output_names = ["prediction_node"]
torch.onnx.export(model, dummy_input, onnx_model_output_path, verbose=True,
                  input_names=input_names, output_names=output_names)

print("\nModel exported successfully!")

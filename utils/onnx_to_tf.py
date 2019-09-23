import onnx
from onnx_tf.backend import prepare


onnx_model_path = "/media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/poseNet/hydraulic/try4(byAnuar)-OndataTry2_withoutAug_sigmaForHeatmaps1InsteadOf2/final_state.onnx"
pb_model_path = "/media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/poseNet/hydraulic/try4(byAnuar)-OndataTry2_withoutAug_sigmaForHeatmaps1InsteadOf2/final_state.pb"
model = onnx.load(onnx_model_path)

# Import the ONNX model to Tensorflow
# without strict=False the graph wouldn't work if there is padding
# in MaxPool layer.
# related: https://github.com/onnx/onnx-tensorflow/issues/167
tf_rep = prepare(model, strict=True, device='CUDA')
tf_rep.export_graph(pb_model_path)


# ===================================================
# Check by loading the .pb model

from load_pb import load_graph
load_graph(pb_model_path)
print("\nSUCCESS!")

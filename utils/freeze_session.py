import tensorflow as tf

"""
Taken from:
https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
"""

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

# yolo = YOLO(backend             = config['model']['backend'],
#             input_size          = config['model']['input_size'], 
#             labels              = config['model']['labels'], 
#             max_box_per_image   = config['model']['max_box_per_image'],
#             anchors             = config['model']['anchors'])

# yolo_inf = yolo.get_inference_model()
# yolo_inf.load_weights(weights_path)
# frozen_graph = freeze_session(K.get_session(),
                      # output_names=[out.op.name for out in yolo_inf.outputs])
# tf.train.write_graph(frozen_graph, "tf_graph", "my_model.pb", as_text=False)

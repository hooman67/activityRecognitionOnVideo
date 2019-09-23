
class LSTMBaseModel(object):
    """
    Base Model from which Activity and Object Detection models inherit.
    Mainly initializes common fields and loads a YOLO model.
    """
    def __init__(self, config, yolo_detector, is_print_model_summary=False):
        self.is_print_model_summary = is_print_model_summary
        self.image_h = config['model']['input_size']
        self.image_w = config['model']['input_size']
        self.stride = config['model']['stride']
        self.h5_sequence_length = config['model']['h5_sequence_length']
        self.last_sequence_length = config['model']['last_sequence_length']
        self.yolo_weights_path = config['model']['detector_weights_path']

        self.sequence_length = self.last_sequence_length / self.stride
        self.num_samples_in_h5 = config['train']['num_samples_in_h5']
        self.lstm_h5_data_path = config['train']['train_h5_folder']

        self.batch_size = config['train']['batch_size']
        self.epochs = config['train']['nb_epochs']
        self.saved_weights_name = config['train']['saved_weights_name']
        self.tensorboard_log_dir = config['train']['tensorboard_log_dir']

        self.detector = yolo_detector
        self.grid_h, self.grid_w = self.detector.grid_h, self.detector.grid_w
        self.debug = config['train']['debug']
        self.initial_epoch = 0

    def load_weights(self, weights_path):
        self.training_model.load_weights(weights_path)
        print("Successfully loaded weights from %s!" % weights_path)

    def freeze_layers(self, model):
        for layer in model.layers:
            layer.trainable = False
        return model

    def set_testing_model_weights(self):
        """
        Training model is fixed in sequence_length, testing model operates
        on each incoming frame online.
        """
        print("\nSetting weights from training model to inference model...")
        for layer_training in self.training_model.layers:
            for layer_testing in self.testing_model.layers:
                if layer_testing.name == layer_training.name:
                    layer_testing.set_weights(layer_training.get_weights())
                    print("set weights for: ", layer_training.name, " == ",
                          layer_testing.name)
                    break

        layer_names = [layer.name for layer in self.training_model.layers]
        layer_testing_names = [layer.name for layer in self.testing_model.layers]
        not_set = set(layer_names) - set(layer_testing_names)
        print("\nLayers for which weights were not set:")
        print([layer_name for layer_name in not_set])
        return

    def load_data_generators_seq(self, batch_size, generator_class, labels):
        # path to folder of H5s is in  self.lstm_h5_data_path
        # is_augment to False because it takes longer time to augment the whole
        # sequence and it's not that important in activity recognition,
        # ok importance in detection though.
        train_batch = generator_class(self.lstm_h5_data_path, batch_size,
                                          self.num_samples_in_h5,
                                          self.last_sequence_length, labels,
                                          stride=self.stride, is_yolo_feats=False,
                                          is_augment=False)
        valid_batch = generator_class(self.lstm_h5_data_path, batch_size,
                                          self.num_samples_in_h5,
                                          self.last_sequence_length, labels,
                                          stride=self.stride, is_validation=True,
                                          is_yolo_feats=False)

        return train_batch, valid_batch

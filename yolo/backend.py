from keras.models import Model
from keras import backend
import tensorflow as tf
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, \
    BatchNormalization, Lambda, SeparableConv2D, DepthwiseConv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.applications.mobilenet import MobileNet
from keras.applications import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

FULL_YOLO_BACKEND_PATH = "full_yolo_backend.h5"  # should be hosted on a server
TINY_YOLO_BACKEND_PATH = "tiny_yolo_backend.h5"  # should be hosted on a server
SQUEEZENET_BACKEND_PATH = "squeezenet_backend.h5"  # should be hosted on a server
MOBILENET_BACKEND_PATH = "mobilenet_backend.h5"  # should be hosted on a server
INCEPTION3_BACKEND_PATH = "inception_backend.h5"  # should be hosted on a server
VGG16_BACKEND_PATH = "vgg16_backend.h5"  # should be hosted on a server
RESNET50_BACKEND_PATH = "resnet50_backend.h5"  # should be hosted on a server


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    """Adds a depthwise convolution block.
    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.
    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating
            the block number.
    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = inputs
    x = DepthwiseConv2D((3, 3),
                       padding='same' if strides == (1, 1) else 'valid',
                       depth_multiplier=depth_multiplier,
                       strides=strides,
                       use_bias=False,
                       name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = LeakyReLU(alpha=0.1, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_id)(x)
    return LeakyReLU(alpha=0.1, name='conv_pw_%d_relu' % block_id)(x)


class BaseFeatureExtractor(object):
    """docstring for ClassName"""

    # to be defined in each subclass
    def __init__(self, input_size):
        raise NotImplementedError("error message")

    # to be defined in each subclass
    def normalize(self, image):
        raise NotImplementedError("error message")

    def get_output_shape(self):
        return self.feature_extractor.get_output_shape_at(-1)[1:3]

    def extract(self, input_image):
        return self.feature_extractor(input_image)


class FullYoloFeature(BaseFeatureExtractor):
    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))

        # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
        def space_to_depth_x2(x):
            import tensorflow as tf
            return tf.space_to_depth(x, block_size=2)

        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1',
                   use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = _depthwise_conv_block(x, 64, 1, block_id=2)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = _depthwise_conv_block(x, 64, 1, block_id=3)
        x = _depthwise_conv_block(x, 128, 1, block_id=4)
        x = _depthwise_conv_block(x, 128, 1, block_id=5)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = _depthwise_conv_block(x, 128, 1, block_id=6)
        x = _depthwise_conv_block(x, 128, 1, block_id=7)
        x = _depthwise_conv_block(x, 256, 1, block_id=8)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = _depthwise_conv_block(x, 256, 1, block_id=9)
        x = _depthwise_conv_block(x, 256, 1, block_id=10)
        x = _depthwise_conv_block(x, 256, 1, block_id=11)
        x = _depthwise_conv_block(x, 512, 1, block_id=12)
        x = _depthwise_conv_block(x, 512, 1, block_id=13)

        skip_connection = x

        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = _depthwise_conv_block(x, 512, 1, block_id=14)
        x = _depthwise_conv_block(x, 512, 1, block_id=15)
        x = _depthwise_conv_block(x, 1024, 1, block_id=16)
        x = _depthwise_conv_block(x, 1024, 1, block_id=17)
        x = _depthwise_conv_block(x, 1024, 1, block_id=18)
        x = _depthwise_conv_block(x, 1024, 1, block_id=19)
        x = _depthwise_conv_block(x, 1024, 1, block_id=20)

        skip_connection = Conv2D(128, (1, 1), strides=(1, 1), padding='same',
                                 name='conv_21', use_bias=False)(skip_connection)
        skip_connection = BatchNormalization(name='norm_21')(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = Lambda(space_to_depth_x2)(skip_connection)

        x = concatenate([skip_connection, x])

        # x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_22',
        #            use_bias=False)(x)
        # x = BatchNormalization(name='norm_22')(x)
        # x = LeakyReLU(alpha=0.1)(x)
        x = _depthwise_conv_block(x, 1024, 1, block_id=22)

        self.feature_extractor = Model(input_image, x)
        # self.feature_extractor.load_weights(FULL_YOLO_BACKEND_PATH)

        self.feature_extractor.summary()
        print('\n')

    def normalize(self, image):
        return image / 255.


class FullYoloFeatureSmall(BaseFeatureExtractor):
    """docstring for ClassName"""

    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))

        # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
        def space_to_depth_x2(x):
            import tensorflow as tf
            return tf.space_to_depth(x, block_size=2)

        # Layer 1
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1',
                   use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(128 / 2, (3, 3), strides=(1, 1), padding='same', name='conv_3',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 4
        x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 5
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(256 / 2, (3, 3), strides=(1, 1), padding='same', name='conv_6',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 7
        x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 9
        x = Conv2D(512 / 2, (3, 3), strides=(1, 1), padding='same', name='conv_9',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 10
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 11
        x = Conv2D(512 / 2, (3, 3), strides=(1, 1), padding='same', name='conv_11',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_11')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 12
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_12')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 13
        x = Conv2D(512 / 2, (3, 3), strides=(1, 1), padding='same', name='conv_13',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_13')(x)
        x = LeakyReLU(alpha=0.1)(x)

        skip_connection = x

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = Conv2D(1024 / 4, (3, 3), strides=(1, 1), padding='same', name='conv_14',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_14')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 15
        x = Conv2D(512 / 2, (1, 1), strides=(1, 1), padding='same', name='conv_15',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_15')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 16
        x = Conv2D(1024 / 4, (3, 3), strides=(1, 1), padding='same', name='conv_16',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_16')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 17
        x = Conv2D(512 / 2, (1, 1), strides=(1, 1), padding='same', name='conv_17',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_17')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 18
        x = Conv2D(1024 / 4, (3, 3), strides=(1, 1), padding='same', name='conv_18',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_18')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 19
        x = Conv2D(1024 / 4, (3, 3), strides=(1, 1), padding='same', name='conv_19',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_19')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 20
        x = Conv2D(1024 / 4, (3, 3), strides=(1, 1), padding='same', name='conv_20',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_20')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 21
        skip_connection = Conv2D(64, (1, 1), strides=(1, 1), padding='same',
                                 name='conv_21', use_bias=False)(skip_connection)
        skip_connection = BatchNormalization(name='norm_21')(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = Lambda(space_to_depth_x2)(skip_connection)

        x = concatenate([skip_connection, x])

        # Layer 22
        x = Conv2D(1024 / 4, (3, 3), strides=(1, 1), padding='same', name='conv_22',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_22')(x)
        x = LeakyReLU(alpha=0.1)(x)

        self.feature_extractor = Model(input_image, x)
        # self.feature_extractor.load_weights(FULL_YOLO_BACKEND_PATH)

        # self.feature_extractor.summary()
        print('\n')

    def normalize(self, image):
        return image / 255.


class TinyYoloFeature(BaseFeatureExtractor):
    """docstring for ClassName"""

    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))

        # Layer 1
        x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='conv_1',
                   use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2 - 5
        for i in range(0, 4):
            x = Conv2D(32 * (2 ** i), (3, 3), strides=(1, 1), padding='same',
                       name='conv_' + str(i + 2), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i + 2))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_6',
                   use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

        # Layer 7 - 8
        for i in range(0, 2):
            x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same',
                       name='conv_' + str(i + 7), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i + 7))(x)
            x = LeakyReLU(alpha=0.1)(x)

        self.feature_extractor = Model(input_image, x)
        self.feature_extractor.load_weights(TINY_YOLO_BACKEND_PATH)

    def normalize(self, image):
        return image / 255.


class MobileNetFeature(BaseFeatureExtractor):
    """docstring for ClassName"""

    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))

        mobilenet = MobileNet(input_shape=(224, 224, 3), include_top=False)
        mobilenet.load_weights(MOBILENET_BACKEND_PATH)

        x = mobilenet(input_image)

        self.feature_extractor = Model(input_image, x)

    def normalize(self, image):
        image = image / 255.
        image = image - 0.5
        image = image * 2.

        return image


class SqueezeNetFeature(BaseFeatureExtractor):
    """docstring for ClassName"""

    def __init__(self, input_size):
        # define some auxiliary variables and the fire module
        sq1x1 = "squeeze1x1"
        exp1x1 = "expand1x1"
        exp3x3 = "expand3x3"
        relu = "relu_"

        def fire_module(x, fire_id, squeeze=16, expand=64):
            s_id = 'fire' + str(fire_id) + '/'

            x = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
            x = Activation('relu', name=s_id + relu + sq1x1)(x)

            left = Conv2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
            left = Activation('relu', name=s_id + relu + exp1x1)(left)

            right = Conv2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
            right = Activation('relu', name=s_id + relu + exp3x3)(right)

            x = concatenate([left, right], axis=3, name=s_id + 'concat')

            return x

        # define the model of SqueezeNet
        input_image = Input(shape=(input_size, input_size, 3))

        x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(input_image)
        x = Activation('relu', name='relu_conv1')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

        x = fire_module(x, fire_id=2, squeeze=16, expand=64)
        x = fire_module(x, fire_id=3, squeeze=16, expand=64)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

        x = fire_module(x, fire_id=4, squeeze=32, expand=128)
        x = fire_module(x, fire_id=5, squeeze=32, expand=128)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

        x = fire_module(x, fire_id=6, squeeze=48, expand=192)
        x = fire_module(x, fire_id=7, squeeze=48, expand=192)
        x = fire_module(x, fire_id=8, squeeze=64, expand=256)
        x = fire_module(x, fire_id=9, squeeze=64, expand=256)

        self.feature_extractor = Model(input_image, x)
        self.feature_extractor.load_weights(SQUEEZENET_BACKEND_PATH)

    def normalize(self, image):
        image = image[..., ::-1]
        image = image.astype('float')

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

        return image


class Inception3Feature(BaseFeatureExtractor):
    """docstring for ClassName"""

    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))

        inception = InceptionV3(input_shape=(input_size, input_size, 3),
                                include_top=False)
        inception.load_weights(INCEPTION3_BACKEND_PATH)

        x = inception(input_image)

        self.feature_extractor = Model(input_image, x)

    def normalize(self, image):
        image = image / 255.
        image = image - 0.5
        image = image * 2.

        return image


class VGG16Feature(BaseFeatureExtractor):
    """docstring for ClassName"""

    def __init__(self, input_size):
        vgg16 = VGG16(input_shape=(input_size, input_size, 3), include_top=False)
        # vgg16.load_weights(VGG16_BACKEND_PATH)

        self.feature_extractor = vgg16

    def normalize(self, image):
        image = image[..., ::-1]
        image = image.astype('float')

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

        return image


class ResNet50Feature(BaseFeatureExtractor):
    """docstring for ClassName"""

    def __init__(self, input_size):
        resnet50 = ResNet50(input_shape=(input_size, input_size, 3), include_top=False)
        resnet50.layers.pop()  # remove the average pooling layer
        # resnet50.load_weights(RESNET50_BACKEND_PATH)

        self.feature_extractor = Model(resnet50.layers[0].input,
                                       resnet50.layers[-1].output)

    def normalize(self, image):
        image = image[..., ::-1]
        image = image.astype('float')

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

        return image

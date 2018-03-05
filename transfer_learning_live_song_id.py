# Imports
import numpy as np
import kapre
from keras.models import (
    Model,
    load_model,
    Sequential,
)
from keras.layers import (
    GlobalAveragePooling2D as GAP2D,
    concatenate,
    Conv2D,
    Reshape,
)
from keras.engine.topology import Input

# Constants
from keras import backend as K
K.set_image_data_format("channels_last")

SR = 22050
DELTA = 8
SAMPLE_STRIDE = 49

BASE_MODEL_FILE = "transfer_learning_music_model.hdf5"
SAMPLE_WIDTH = int(5.12 * 16000)
NUM_FEATURES = 160

# Helper Models
def build_feature_extractor():
    """Builds the transfer_learning_music feature extractor."""
    base_model = load_model(
        BASE_MODEL_FILE,
        custom_objects={
            "Melspectrogram": kapre.time_frequency.Melspectrogram,
            "Normalization2D": kapre.utils.Normalization2D,
        },
    )

    feat_layer1 = GAP2D()(base_model.get_layer("elu_1").output)
    feat_layer2 = GAP2D()(base_model.get_layer("elu_2").output)
    feat_layer3 = GAP2D()(base_model.get_layer("elu_3").output)
    feat_layer4 = GAP2D()(base_model.get_layer("elu_4").output)
    feat_layer5 = GAP2D()(base_model.get_layer("elu_5").output)

    feat_all = concatenate([feat_layer1, feat_layer2, feat_layer3, feat_layer4, feat_layer5])

    return Model(inputs=base_model.input, outputs=feat_all)

def build_delta(num_samples):
    """Builds the delta convolution model."""
    delta_width = DELTA + 1
    num_filters = num_samples - delta_width

    delta_vec = np.zeros(delta_width)
    delta_vec[0] = 1
    delta_vec[-1] = -1

    delta_ker = np.zeros((delta_width, 1, 1, num_filters))
    for i in range(num_filters):
        delta_ker[:,0,0,i] = delta_vec

    def delta_ker_init(shape, dtype=None):
        assert delta_ker.shape == shape, (delta_ker.shape, shape)
        return delta_ker

    return Sequential([
        Reshape(
            input_shape=(NUM_FEATURES, num_samples),
            target_shape=(NUM_FEATURES, num_samples, 1),
        ),
        Conv2D(
            filters=num_filters,
            kernel_size=(delta_width, 1),
            kernel_initializer=delta_ker_init,
            use_bias=False,
            padding="valid",
        ),
    ])

# Preprocessing
def get_num_samples(audio_len):
    """Get the number of samples to take."""
    return (audio_len - SAMPLE_WIDTH)//SAMPLE_STRIDE

def get_samples(audio_arr):
    """Sample the given audio."""
    audio_len, = audio_arr.shape
    num_samples = get_num_samples(audio_len)

    samples = np.zeros((num_samples, SAMPLE_WIDTH))
    for i in range(0, num_samples, SAMPLE_STRIDE):
        samples[i] = audio_arr[i:i+SAMPLE_WIDTH+1]
    return samples

# Main Model
def build_model(audio_len=6*SR):
    """Build the combined feature extraction and delta model."""
    num_samples = get_num_samples(audio_len)

    input_layer = Input(shape=(1, SAMPLE_WIDTH))

    feat_model = build_feature_extractor()
    print(input_layer, "->", feat_model.input)
    feat_layer = feat_model(input_layer)

    delta_model = build_delta(num_samples)
    print(feat_model.output, "->", delta_model.input)
    delta_layer = delta_model(feat_layer)

    return Model(inputs=input_layer, outputs=delta_layer)

if __name__ == "__main__":
    build_model().summary()

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
SAMPLE_STRIDE = 98

BASE_MODEL_FILE = "transfer_learning_music_model.hdf5"
SAMPLE_WIDTH = int(5.12 * 16000)
NUM_FEATURES = 160

# Base Models
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
            input_shape=(num_samples, NUM_FEATURES),
            target_shape=(num_samples, NUM_FEATURES, 1),
        ),
        Conv2D(
            filters=num_filters,
            kernel_size=(delta_width, 1),
            kernel_initializer=delta_ker_init,
            use_bias=False,
            padding="valid",
        ),
    ])

# Utilities
def get_num_samples(audio_len):
    """Get the number of samples to take."""
    return (audio_len - SAMPLE_WIDTH)//SAMPLE_STRIDE

def get_samples(audio_arr):
    """Sample the given audio."""
    audio_len, = audio_arr.shape
    num_samples = get_num_samples(audio_len)

    samples = np.zeros((num_samples, 1, SAMPLE_WIDTH))
    for i in range(0, num_samples):
        j = i * SAMPLE_STRIDE
        samples[i,0] = audio_arr[j:j+SAMPLE_WIDTH]
    return samples

def binary_threshold(delta_arr, threshold=0):
    """Cast the given array to binary."""
    return np.where(delta_arr >= threshold, 1, 0)

def predict_all(samples, feat_extractor, max_batch_size=128):
    """Split the samples into smaller batches to save memory."""
    num_samples = samples.shape[0]

    batch_indices = list(range(max_batch_size, num_samples, max_batch_size))
    batches = np.split(samples, batch_indices)

    features = np.zeros((num_samples, NUM_FEATURES))
    for i, batch in enumerate(batches):
        batch_size = batch.shape[0]
        j = i * max_batch_size
        features[j:j+batch_size] = feat_extractor.predict(batch, batch_size=batch_size)
    return features

# Main Model
def build_models(audio_len=6*SR):
    """Build the combined feature extraction and delta model."""
    num_samples = get_num_samples(audio_len)
    return build_feature_extractor(), build_delta(num_samples)

def run_models(audio_arr, feat_extractor, delta_model):
    """Run the given models on the given audio."""
    samples = get_samples(audio_arr)
    num_samples = samples.shape[0]

    features = predict_all(samples, feat_extractor)
    assert features.shape == (num_samples, NUM_FEATURES), (features.shape, (num_samples, NUM_FEATURES))
    features = features.reshape((1, num_samples, NUM_FEATURES))

    delta_arr = delta_model.predict(features, batch_size=1)
    binary_arr = binary_threshold(delta_arr)
    return binary_arr

# Testing
if __name__ == "__main__":
    audio_len = 6*SR
    test_audio = np.random.random(audio_len)

    models = build_models(audio_len)
    for model in models:
        model.summary()

    hashprint = run_models(test_audio, *models)
    print(hashprint.shape)
    print(hashprint)

# Imports
from keras.models import Model, load_model
from keras.layers import GlobalAveragePooling2D as GAP2D
from keras.layers import concatenate

import kapre

# Constants
BASE_MODEL_FILE = "transfer_learning_music_model.hdf5"

# Transfer Learning Music
def build_feature_extractor():
    """Builds the transfer_learning_music feature extractor."""
    base_model = load_model(
        BASE_MODEL_FILE,
        custom_objects={
            "Melspectrogram":kapre.time_frequency.Melspectrogram,
            "Normalization2D":kapre.utils.Normalization2D,
        },
    )

    feat_layer1 = GAP2D()(base_model.get_layer("elu_1").output)
    feat_layer2 = GAP2D()(base_model.get_layer("elu_2").output)
    feat_layer3 = GAP2D()(base_model.get_layer("elu_3").output)
    feat_layer4 = GAP2D()(base_model.get_layer("elu_4").output)
    feat_layer5 = GAP2D()(base_model.get_layer("elu_5").output)

    feat_all = concatenate([feat_layer1, feat_layer2, feat_layer3, feat_layer4, feat_layer5])

    return Model(inputs=base_model.input, outputs=feat_all)

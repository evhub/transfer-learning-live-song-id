# Imports
import sys
import os
import math
import time

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

try:
    import song_db
except SyntaxError:
    import song_db2 as song_db

sys.path.append(os.path.dirname(__file__))
from search import calculateMRR

# Constants
from keras import backend as K
K.set_image_data_format("channels_last")

SR = 22050
DELTA = 8
SAMPLE_STRIDE = 1024

BASE_MODEL_FILE = "transfer_learning_music_model.hdf5"
SAMPLE_WIDTH = int(5.12 * 16000)
NUM_FEATURES = 160

DB_DIR = os.path.join(os.path.dirname(__file__), "db")
REFS_DIR = os.path.join(DB_DIR, "refs")
QUERIES_DIR = os.path.join(DB_DIR, "queries")

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

FEAT_EXTRACTOR = build_feature_extractor()

def build_delta(num_samples):
    """Builds the delta convolution model."""
    delta_width = DELTA + 1

    delta_vec = np.zeros(delta_width)
    delta_vec[0] = 1
    delta_vec[-1] = -1

    delta_ker = delta_vec.reshape((delta_width, 1, 1, 1))
    def delta_ker_init(shape, dtype=None):
        assert delta_ker.shape == shape, (delta_ker.shape, shape)
        return delta_ker

    return Sequential([
        Reshape(
            input_shape=(num_samples, NUM_FEATURES),
            target_shape=(num_samples, NUM_FEATURES, 1),
        ),
        Conv2D(
            filters=1,
            kernel_size=(delta_width, 1),
            kernel_initializer=delta_ker_init,
            use_bias=False,
            padding="valid",
        ),
    ])

# Utilities
def get_num_samples(audio_len):
    """Get the number of samples to take."""
    remaining_len = audio_len - SAMPLE_WIDTH + 1
    if remaining_len <= 0:
        return None
    return audio_len//SAMPLE_STRIDE

def get_samples(audio_arr):
    """Sample the given audio."""
    audio_len, = audio_arr.shape
    num_samples = get_num_samples(audio_len)
    assert num_samples, num_samples

    samples = np.zeros((num_samples, 1, SAMPLE_WIDTH))
    for i in range(0, num_samples):
        j = i * SAMPLE_STRIDE
        sample = audio_arr[j:j+SAMPLE_WIDTH]
        n, = sample.shape
        samples[i,0,:n] = sample
    return samples

def binary_threshold(delta_arr, threshold=0):
    """Cast the given array to binary."""
    return np.where(delta_arr >= threshold, 1, 0)

def predict_all(samples, feat_extractor, ideal_batch_size=128, max_batch_size=192):
    """Split the samples into smaller batches to save memory."""
    num_samples = samples.shape[0]

    if num_samples <= max_batch_size:
        batches = [samples]
    else:
        batch_indices = list(range(ideal_batch_size, num_samples, ideal_batch_size))
        batches = np.split(samples, batch_indices)

    features = np.zeros((num_samples, NUM_FEATURES))
    for i, batch in enumerate(batches):
        batch_size = batch.shape[0]
        j = i * ideal_batch_size
        features[j:j+batch_size] = feat_extractor.predict(batch, batch_size=batch_size)
    return features

# Main Model
def build_models(audio_len):
    """Build the combined feature extraction and delta model."""
    num_samples = get_num_samples(audio_len)
    assert num_samples, num_samples
    return FEAT_EXTRACTOR, build_delta(num_samples)

def run_models(audio_arr, feat_extractor, delta_model):
    """Run the given models on the given audio."""
    samples = get_samples(audio_arr)
    num_samples = samples.shape[0]

    features = predict_all(samples, feat_extractor)
    assert features.shape == (num_samples, NUM_FEATURES), (features.shape, (num_samples, NUM_FEATURES))
    features = features.reshape((1, num_samples, NUM_FEATURES))

    delta_arr = delta_model.predict(features, batch_size=1)
    binary_arr = binary_threshold(delta_arr)

    out_arr = np.squeeze(binary_arr).T
    assert out_arr.shape[0] == NUM_FEATURES, out_arr.shape
    return out_arr

def process(audio_arr, debug=False):
    """Build and run models on the given audio."""
    audio_len, = audio_arr.shape
    num_samples = get_num_samples(audio_len)
    if debug:
        print("\tProcessing audio array of length %r (%r samples)..." % (audio_len, num_samples))
        t0 = time.clock()
    models = build_models(audio_len)
    result = run_models(audio_arr, *models)
    if debug:
        tf = time.clock()
        print("\t\t(took %rs)" % (tf - t0,))
    return result

def process_all(audio_arrs, debug=False):
    """Process all the given audio arrays."""
    return [process(audio, debug) for audio in audio_arrs]

# Testing
if __name__ == "__main__":
    print("Testing models...")
    audio_len = SAMPLE_WIDTH
    test_audio = np.random.random(audio_len)
    bin_repr = process(test_audio, debug=True)
    print(bin_repr.shape)
    print(bin_repr)

# Database management
def make_db():
    """Create all the db directories if they need to be made."""
    made_dir = False
    for dirpath in [DB_DIR, REFS_DIR, QUERIES_DIR]:
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
            made_dir = True
    return made_dir

def get_ref_path(index):
    """Get the path to the processed reference of the given index."""
    return os.path.join(REFS_DIR, "{}.npy".format(index))

def get_query_path(index):
    """Get the path to the processed query of the given index."""
    return os.path.join(QUERIES_DIR, "{}.npy".format(index))

def write_db(proc_refs, proc_queries):
    """Writes processed refs and queries to the database."""
    for i, ref in enumerate(proc_refs):
        ref_path = get_ref_path(i)
        np.save(ref_path, ref)
    for i, query in enumerate(proc_queries):
        query_path = get_query_path(i)
        np.save(query_path, query)

def sorted_paths(paths):
    """Sorts file paths by their number."""
    return sorted(paths, key=lambda p: int(p.split(".", 1)[0]))

def read_db(debug=False):
    """Reads processed refs and queries from the database."""
    refs = []
    for ref_name in sorted_paths(os.listdir(REFS_DIR)):
        if debug:
            print("\tLoading ref %s..." % (ref_name,))
        ref_path = os.path.join(REFS_DIR, ref_name)
        refs.append(np.load(ref_path))
    queries = []
    for query_name in sorted_paths(os.listdir(QUERIES_DIR)):
        if debug:
            print("\tLoading query %s..." % (query_name,))
        query_path = os.path.join(QUERIES_DIR, query_name)
        queries.append(np.load(query_path))
    return refs, queries

def remove_short_queries(queries, groundTruth):
    """Removes queries that are too short from queries and groundTruth."""
    assert len(queries) == len(groundTruth), (len(queries), len(groundTruth))
    filt_queries = []
    filt_groundTruth = []
    for query, truth in zip(queries, groundTruth):
        audio_len, = query.shape
        num_samples = get_num_samples(audio_len)
        if num_samples is not None:
            filt_queries.append(query)
            filt_groundTruth.append(truth)
    assert len(filt_queries) == len(filt_groundTruth), (len(filt_queries), len(filt_groundTruth))
    return filt_queries, np.asarray(filt_groundTruth)

# Calculating MRR
if __name__ == "__main__":
    print("Querying database...")
    refs, queries, groundTruth = song_db.get_data_for_artist("taylorswift")
    queries, groundTruth = remove_short_queries(queries, groundTruth)
    if make_db():
        print("Processing refs...")
        proc_refs = process_all(refs, debug=True)
        print("Processing queries...")
        proc_queries = process_all(queries, debug=True)
        print("Writing to database...")
        write_db(proc_refs, proc_queries)
    else:
        print("Using existing database...")
        proc_refs, proc_queries = read_db(debug=True)
    print("Calculating MRR for %r refs and %r queries..." % (len(proc_refs), len(proc_queries)))
    MMR = calculateMRR(proc_queries, proc_refs, groundTruth)
    print("MMR =", MMR)

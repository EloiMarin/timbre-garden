"""03. Corpus-based analysis and synthesis
"""

import librosa
import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.decomposition
import sklearn.preprocessing

import taichi as ti
from tolvera import Tolvera, run
#from tolvera.osc.update import Updater
# from tolvera.utils import ti_map_range
from functools import reduce
from operator import add
from math import floor

from threading import Timer

def compute_mfccs(buffer, sample_rate, fft_size, hop_size, n_mfcc):
    """
    Extract MFCC, and rescale to zero-mean, unit-variance.
    FluCoMa has a nice interactive explainer that gives some intuition:
    https://learn.flucoma.org/reference/mfcc/
    """
    print("Computing mfcss...")
    X = librosa.feature.mfcc(y=buffer.data[0], sr=sample_rate, n_fft=fft_size, hop_length=hop_size, n_mfcc=20)
    X = sklearn.preprocessing.scale(X)
    X = X.T
    print("MFCC coefficient shape: %s" % str(X.shape))
    print(np.round(X[:8,:8], 4))
    return X

def perform_pca(X, n_components, whiten):
    """
    Perform Principal Component Analysis (PCA) to reduce the dimensionality of
    each input MFCC frame, and extract various manually-specified features.
    """
    model = sklearn.decomposition.PCA(n_components=2, whiten=True)
    model.fit(X)
    Y = model.transform(X)
    print("Y shape: %s" % str(Y.shape))
    return Y

def create_data_series(Y, sample_rate, fft_size, hop_size):
    """
    Create data series, containing per-segment properties which are later needed
    for display and playback:
    - ordinal index
    - timestamp (in seconds)
    - duration (the same for every block, as we're using identically-sized blocks)
    """
    index = np.arange(len(Y))
    timestamp = index * hop_size / sample_rate
    duration = 100 * fft_size / sample_rate
    duration_array = np.array([duration] * len(Y))
    return index, timestamp, duration_array

def floats_to_ordinals(floats):
    sorted_indices = np.argsort(floats)
    positions = np.argsort(sorted_indices)
    return positions.tolist()

def extract_audio_features(buffer, Y, fft_size, hop_size):
    """
    Manually extract a few features to add to the data frame:
    - spectral centroid
    - spectral flatness
    - k-means cluster
    """
    centroid = librosa.feature.spectral_centroid(y=buffer.data[0], sr=buffer.sample_rate, n_fft=fft_size, hop_length=hop_size)[0]
    flatness = librosa.feature.spectral_flatness(y=buffer.data[0], n_fft=fft_size, hop_length=hop_size)[0]
    flatness = floats_to_ordinals(flatness)
    kmeans = sklearn.cluster.KMeans(n_clusters=4)
    labels = kmeans.fit_predict(Y)
    return centroid, flatness, labels

def create_features_df(Y, centroid, flatness, index, timestamp, duration_array, labels):
    """
    Aggregate all features into a pandas DataFrame, 
    with columns for each features and a row for each segment
    """
    return pd.DataFrame({
        "x": Y[:,0],
        "y": Y[:,1],
        "centroid": centroid,
        "flatness": flatness,
        "timestamp": timestamp/timestamp.max(), # normalize
        "duration": duration_array/timestamp.max(), # normalize
        "cluster": labels,
    })

def load_audio(filename):
    """
    Load an audio file from disk and return a buffer
    """
    pass
    #return Buffer(filename)

def analyse_audio(buffer, fft_size=16384, n_mfcc=20, n_components=2):
    """
    This method demonstrates how to:
    - extract features from a pre-recorded corpus of audio (here, a large audio file)
    - perform dimensionality reduction
    - transform feature arrays into a pandas dataframe
    """
    hop_size = fft_size // 2
    sample_rate = buffer.sample_rate

    print("Analyzing audio...")
    X = compute_mfccs(buffer, sample_rate, fft_size, hop_size, n_mfcc)
    Y = perform_pca(X, n_components, True)
    index, timestamp, duration_array = create_data_series(Y, sample_rate, fft_size, hop_size)
    centroid, flatness, labels = extract_audio_features(buffer, Y, fft_size, hop_size)
    df = create_features_df(Y, centroid, flatness, index, timestamp, duration_array, labels)
    return df

def save_features_df(df, filename="features.csv"):
    print("Saving features to %s" % filename)
    df.to_csv(filename, index=False)

def load_features_df(filename="features.csv"):
    print("Loading features from %s" % filename)
    return pd.read_csv(filename)

def df_to_np_dict(df):
    dict_of_lists = df.to_dict(orient='list')
    dict_of_arrays = {key: np.array(value).astype(np.float32) for key, value in dict_of_lists.items()}
    return dict_of_arrays

def prepare_attraction_points(df):
    """
    Prepare the attraction points based on the centroids of the audio features.
    """
    centroids = df['centroid'].values
    normalized_centroids = (centroids - centroids.min()) / (centroids.max() - centroids.min())
    positions = np.stack((df['x'].values, normalized_centroids), axis=-1)
    masses = np.ones_like(centroids)  # Mass of 1
    return positions.astype(np.float32), masses.astype(np.float32)

def main(**kwargs):
    tv = Tolvera(**kwargs)

    audio_file = kwargs.get("audio_file", "qmul-workshop/audio/sunkilmoon-truckers-atlas-loop.wav")
    analyse = kwargs.get("analyse", False)
    load = kwargs.get("load", False)
    
    df = None
    if analyse:
        buffer = load_audio(audio_file)
        df = analyse_audio(buffer)
        save_features_df(df)
    elif load:
        df = load_features_df()

    np_dict = df_to_np_dict(df)

    tv.s.grains = {
        'state': {
            'x': (ti.f32, -3., 3.),
            'y': (ti.f32, -3., 3.),
            'centroid': (ti.f32, 0., 20000.),
            'flatness': (ti.f32, 0., 100.),
            'timestamp': (ti.f32, 0., 1.),
            'duration': (ti.f32, 0., 1.),
            'cluster': (ti.i32, 0., 1.),
        },
        'shape': (len(df)),
    }
    tv.s.grains.set_from_nddict(np_dict)

    @ti.func
    def ti_map_range(val, in_min, in_max, out_min, out_max):
        return out_min + ((val - in_min) * (out_max - out_min)) / (in_max - in_min)
    
    @ti.func
    def grain_pos_f32(i):
        """Get a grain's position in pixels as a float."""
        x, y = tv.s.grains[i].x, tv.s.grains[i].y
        x, y = ti_map_range(x, -3., 3., 0., tv.x), ti_map_range(y, -3., 3., 0., tv.y)
        x, y = ti.cast(x, ti.f32), ti.cast(y, ti.f32)
        return x, y
    
    @ti.func
    def grain_pos_i32(i):
        """Get a grain's position in pixels as an integer."""
        x, y = grain_pos_f32(i)
        x, y = ti.cast(x, ti.i32), ti.cast(y, ti.i32)
        return x, y

    @ti.kernel
    def draw_grains():
        """
        As in the original notebook, we map timestamp to colour.
        """
        for i in range(tv.s.grains.shape[0]):
            x, y = grain_pos_i32(i)
            t = tv.s.grains[i].timestamp
            rgba = ti.Vector([t, 0.5, 0.5, 1.0])
            tv.px.circle(x, y, 10, rgba)

    @ti.kernel
    def attract_grains(radius: ti.f32):
        for gi, pi in ti.ndrange(tv.s.grains.shape[0], tv.p.field.shape[0]):
            g = ti.Vector(grain_pos_f32(gi))
            p = tv.p.field[pi]
            dist = p.pos - g
            if dist.norm() < radius:
                tv.v._attract(tv.p, g, 1, tv.x)

    def RaveOSC():
        collision_field = ti.field(ti.i32, shape=tv.s.grains.shape[0])
        collision_field.fill(0)

        @ti.kernel
        def detect_collisions(radius: ti.f32):
            for gi, pi in ti.ndrange(tv.s.grains.shape[0], tv.p.field.shape[0]):
                collision_field[gi] = 0
                g = ti.Vector(grain_pos_f32(gi))
                p = tv.p.field[pi]
                if p.active == 0:
                    continue
                dist = p.pos - g
                if dist.norm() < radius:
                    collision_field[gi] = 1

        @tv.osc.map.send_list(vector=(0.,0.,100.), count=1, length=16, send_mode="broadcast")
        def rave_dimensions() -> list[float]:
            detect_collisions(10)
            np_collision_field = collision_field.to_numpy()

            message = []
            dimension_size = floor(len(np_collision_field) / 16)
            for dim in range(0, 16):
                base = dim * dimension_size
                message.append(reduce(add, np_collision_field[base:base + dimension_size]))

            return map(float, message)

        @tv.osc.map.receive_args(arg=(0,1,1), count=1)
        def tolvera_reset(args):
            if (args != 0):
                tv.randomise()

    RaveOSC()

    @tv.render
    def _():
        tv.px.diffuse(0.99)
        tv.v.flock(tv.p)
        draw_grains()
        attract_grains(10)
        tv.px.particles(tv.p, tv.s.species())
        return tv.px

if __name__ == '__main__':
    run(main)

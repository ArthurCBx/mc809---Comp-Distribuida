import keras
import tensorflow as tf

@keras.saving.register_keras_serializable(package='siamese')
def euclidean_distance(vectors):
    x, y = vectors
    return tf.norm(x - y, ord='euclidean', axis=1, keepdims=True)
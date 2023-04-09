import tensorflow as tf
import numpy as np
from models.models import l2_regularizer


def linear(embeddings, labels, num_classes, scale=1.0, margin=0.0, kernel_initializer=tf.compat.v1.orthogonal_initializer(), kernel_regularizer=l2_regularizer(1e-3), name="linear"):
    emb_shape = embeddings.get_shape().as_list()
    assert len(emb_shape) == 2
    emb_dim = emb_shape[1]

    with tf.variable_scope(name_or_scope=None, default_name=name) as scope:
        kernel = tf.get_variable(name="kernel", shape=[emb_dim, num_classes], dtype=embeddings.dtype,
                                 initializer=kernel_initializer,
                                 regularizer=kernel_regularizer)
        logits = tf.matmul(embeddings, kernel)
        return logits


def am_linear(embeddings, labels, num_classes, scale=32.0, margin=0.3, kernel_initializer=tf.compat.v1.orthogonal_initializer(), kernel_regularizer=l2_regularizer(1e-3), name="am_linear"):
    emb_shape = embeddings.get_shape().as_list()
    assert len(emb_shape) == 2
    emb_dim = emb_shape[1]

    with tf.variable_scope(name_or_scope=None, default_name=name) as scope:
        kernel = tf.get_variable(name="kernel", shape=[emb_dim, num_classes], dtype=embeddings.dtype,
                                 initializer=kernel_initializer,
                                 regularizer=kernel_regularizer)

        embeddings_norm = tf.nn.l2_normalize(embeddings, 1, 1e-5)
        kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-5)

        cos_theta = tf.matmul(embeddings_norm, kernel_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)

        labels_onehot = tf.cast(tf.one_hot(labels, num_classes), embeddings.dtype)
        logits = cos_theta - margin * labels_onehot
        scaled_logits = scale * logits
        return scaled_logits


def aam_linear(embeddings, labels, num_classes, scale=32.0, margin=0.3, kernel_initializer=tf.compat.v1.orthogonal_initializer(), kernel_regularizer=l2_regularizer(1e-3), name="aam_linear"):
    emb_shape = embeddings.get_shape().as_list()
    assert len(emb_shape) == 2
    emb_dim = emb_shape[1]

    with tf.variable_scope(name_or_scope=None, default_name=name) as scope:
        kernel = tf.get_variable(name="kernel", shape=[emb_dim, num_classes], dtype=embeddings.dtype,
                                 initializer=kernel_initializer,
                                 regularizer=kernel_regularizer)

        embeddings_norm = tf.nn.l2_normalize(embeddings, 1, 1e-5)
        kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-5)

        cos_theta = tf.matmul(embeddings_norm, kernel_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)
        sin_theta = tf.sqrt(1 - cos_theta * cos_theta)
        phi = cos_theta * tf.cos(margin) - sin_theta * tf.sin(margin) - 0.5 * margin * margin

        labels_onehot = tf.cast(tf.one_hot(labels, num_classes), embeddings.dtype)
        logits = phi * labels_onehot + cos_theta * (1 - labels_onehot)
        scaled_logits = scale * logits
        return scaled_logits


def cm_linear(embeddings, labels, num_classes, scale=32.0, margin=0.2, kernel_initializer=tf.compat.v1.orthogonal_initializer(), kernel_regularizer=l2_regularizer(1e-3), name="cm_linear"):
    emb_shape = embeddings.get_shape().as_list()
    assert len(emb_shape) == 2
    emb_dim = emb_shape[1]

    margin_0 = margin
    margin_1 = 0.5 * margin * margin

    with tf.variable_scope(name_or_scope=None, default_name=name) as scope:
        kernel = tf.get_variable(name="kernel", shape=[emb_dim, num_classes], dtype=embeddings.dtype,
                                 initializer=kernel_initializer,
                                 regularizer=kernel_regularizer)

        embeddings_norm = tf.nn.l2_normalize(embeddings, 1, 1e-5)
        kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-5)

        cos_theta = tf.matmul(embeddings_norm, kernel_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)
        sin_theta = tf.sqrt(1 - cos_theta * cos_theta)
        # phi = cos_theta * tf.cos(margin[0]) - sin_theta * tf.sin(margin[0]) - margin[1]
        phi = cos_theta * tf.cos(margin_0) - sin_theta * tf.sin(margin_0) - margin_1

        labels_onehot = tf.cast(tf.one_hot(labels, num_classes), embeddings.dtype)
        logits = phi * labels_onehot + cos_theta * (1 - labels_onehot)
        scaled_logits = scale * logits
        return scaled_logits


def cm_linear_voxsrc2020(embeddings, labels, num_classes, scale=32.0, margin=0.2, kernel_initializer=tf.compat.v1.orthogonal_initializer(), kernel_regularizer=l2_regularizer(1e-3), name="cm_linear_voxsrc2020"):
    emb_shape = embeddings.get_shape().as_list()
    assert len(emb_shape) == 2
    emb_dim = emb_shape[1]

    margin_0 = margin
    margin_1 = margin / 2.0

    with tf.variable_scope(name_or_scope=None, default_name=name) as scope:
        kernel = tf.get_variable(name="kernel", shape=[emb_dim, num_classes], dtype=embeddings.dtype,
                                 initializer=kernel_initializer,
                                 regularizer=kernel_regularizer)

        embeddings_norm = tf.nn.l2_normalize(embeddings, 1, 1e-5)
        kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-5)

        cos_theta = tf.matmul(embeddings_norm, kernel_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)
        sin_theta = tf.sqrt(1 - cos_theta * cos_theta)
        phi = cos_theta * tf.cos(margin_0) - sin_theta * tf.sin(margin_0) - margin_1

        labels_onehot = tf.cast(tf.one_hot(labels, num_classes), embeddings.dtype)
        logits = phi * labels_onehot + cos_theta * (1 - labels_onehot)
        scaled_logits = scale * logits
        return scaled_logits


def hcm_linear(embeddings, labels, num_classes, scale=32.0, margin=(0.2, 0.1), kernel_initializer=tf.compat.v1.orthogonal_initializer(), kernel_regularizer=l2_regularizer(1e-3), name="hcm_linear"):
    hard_margin = 0.1
    emb_shape = embeddings.get_shape().as_list()
    # emb_shape = tf.shape(embeddings)
    assert len(emb_shape) == 2
    emb_dim = emb_shape[1]

    with tf.variable_scope(name_or_scope=None, default_name=name) as scope:
        kernel = tf.get_variable(name="kernel", shape=[emb_dim, num_classes], dtype=embeddings.dtype,
                                 initializer=kernel_initializer,
                                 regularizer=kernel_regularizer)

        embeddings_norm = tf.nn.l2_normalize(embeddings, 1, 1e-5)
        kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-5)

        cos_theta = tf.matmul(embeddings_norm, kernel_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)
        sin_theta = tf.sqrt(1 - cos_theta * cos_theta)
        phi = cos_theta * tf.cos(margin[0]) - sin_theta * tf.sin(margin[0]) - margin[1]

        labels_onehot = tf.cast(tf.one_hot(labels, num_classes), embeddings.dtype)

        batch_size = tf.shape(labels)[0]
        column_range = tf.range(batch_size)
        indices = tf.stack([column_range, labels], axis=1)
        target_phi = tf.reshape(tf.gather_nd(phi, indices), [-1, 1])

        hard_mask = tf.cast(cos_theta > target_phi, embeddings.dtype)

        logits = phi * labels_onehot + \
            (cos_theta * (1 - hard_mask) + (cos_theta + hard_margin) * hard_mask) * (1 - labels_onehot)
        scaled_logits = scale * logits
        return scaled_logits


# def sc_cm_linear(embeddings, labels, num_classes, scale=32.0, margin=(0.2, 0.1), num_centers=2, kernel_initializer=tf.compat.v1.orthogonal_initializer(), kernel_regularizer=l2_regularizer(1e-3), name="sc_cm_linear"):
def sc_cm_linear(embeddings, labels, num_classes, scale=32.0, margin=0.2, num_centers=2, kernel_initializer=tf.compat.v1.orthogonal_initializer(), kernel_regularizer=l2_regularizer(1e-3), name="sc_cm_linear"):
    emb_shape = embeddings.get_shape().as_list()
    assert len(emb_shape) == 2
    emb_dim = emb_shape[1]

    margin_0 = margin
    margin_1 = 0.5 * margin * margin

    with tf.variable_scope(name_or_scope=None, default_name=name) as scope:
        kernel = tf.get_variable(name="kernel", shape=[num_centers, emb_dim, num_classes], dtype=embeddings.dtype,
                                 initializer=kernel_initializer,
                                 regularizer=kernel_regularizer)

        embeddings_norm = tf.nn.l2_normalize(embeddings, 1, 1e-5)
        # kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-5)
        kernel_norm = tf.nn.l2_normalize(kernel, 1, 1e-5)

        # cos_theta = tf.matmul(embeddings_norm, kernel_norm)
        cos_theta = tf.reduce_max(tf.matmul(embeddings_norm, kernel_norm), 0, keepdims=False)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)
        sin_theta = tf.sqrt(1 - cos_theta * cos_theta)
        # phi = cos_theta * tf.cos(margin[0]) - sin_theta * tf.sin(margin[0]) - margin[1]
        phi = cos_theta * tf.cos(margin_0) - sin_theta * tf.sin(margin_0) - margin_1

        labels_onehot = tf.cast(tf.one_hot(labels, num_classes), embeddings.dtype)
        logits = phi * labels_onehot + cos_theta * (1 - labels_onehot)
        scaled_logits = scale * logits
        return scaled_logits


def sc_am_linear(embeddings, labels, num_classes, scale=32.0, margin=0.2, num_centers=2, kernel_initializer=tf.compat.v1.orthogonal_initializer(), kernel_regularizer=l2_regularizer(1e-3), name="sc_am_linear"):
    emb_shape = embeddings.get_shape().as_list()
    assert len(emb_shape) == 2
    emb_dim = emb_shape[1]

    with tf.variable_scope(name_or_scope=None, default_name=name) as scope:
        kernel = tf.get_variable(name="kernel", shape=[num_centers, emb_dim, num_classes], dtype=embeddings.dtype,
                                 initializer=kernel_initializer,
                                 regularizer=kernel_regularizer)

        embeddings_norm = tf.nn.l2_normalize(embeddings, 1, 1e-5)

        kernel_norm = tf.nn.l2_normalize(kernel, 1, 1e-5)

        cos_theta = tf.reduce_max(tf.matmul(embeddings_norm, kernel_norm), 0, keepdims=False)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)


        labels_onehot = tf.cast(tf.one_hot(labels, num_classes), embeddings.dtype)
        logits = cos_theta - margin * labels_onehot

        scaled_logits = scale * logits
        return scaled_logits


def qm_linear(embeddings, labels, num_classes, scale=32.0, margin=0.3, kernel_initializer=tf.compat.v1.orthogonal_initializer(), kernel_regularizer=l2_regularizer(1e-3), name="qm_linear"):
    emb_shape = embeddings.get_shape().as_list()
    assert len(emb_shape) == 2
    emb_dim = emb_shape[1]

    with tf.variable_scope(name_or_scope=None, default_name=name) as scope:
        delta = (1 - margin) / 2
        kernel = tf.get_variable(name="kernel", shape=[emb_dim, num_classes], dtype=embeddings.dtype,
                                 initializer=kernel_initializer,
                                 regularizer=kernel_regularizer)

        embeddings_norm = tf.nn.l2_normalize(embeddings, 1, 1e-5)
        kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-5)

        cos_theta = tf.matmul(embeddings_norm, kernel_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)

        labels_onehot = tf.cast(tf.one_hot(labels, num_classes), embeddings.dtype)
        logits = (cos_theta - (1 - delta)) * ((1 + delta) - cos_theta) * labels_onehot + (cos_theta - delta) * (cos_theta + delta) * (1 - labels_onehot)
        scaled_logits = scale * logits
        return scaled_logits

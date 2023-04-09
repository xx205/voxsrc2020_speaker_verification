import argparse
import importlib

import tensorflow as tf
import horovod.tensorflow as hvd
import tf_projection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module_source', type=str, dest="module_source",
                        help='module source file, e.g. models.tdnn_model')
    parser.add_argument('--model_id', type=str, dest="model_id",
                        help='model id defined in model source, e.g. tdnn')
    parser.add_argument("--checkpoint_directory", type=str, dest="checkpoint_directory",
                        help="checkpoint directory")
    parser.add_argument("--expansion_dim", type=int, dest="expansion_dim", default=2,
                        help="expansion dim, 2 for 1D conv models like tdnn (NHWC, W=1), 3 for 2D conv models like res2net (NHWC, C=1)")
    parser.add_argument("--feat_dim", type=int, dest="feat_dim", default=40,
                        help="feature dimension, e.g. 40 for 40-dimensional FBANK")
    args = parser.parse_args()

    model = getattr(importlib.import_module(args.module_source), args.model_id)

    input_dims = [None, None, args.feat_dim]
    input_dims.insert(args.expansion_dim, 1)

    X = tf.placeholder(tf.float32, input_dims, name="inputs")

    try:
        x = model(X, training=False)
    except:
        x = model(X)

    init = tf.global_variables_initializer()

    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    # tvars = tf.trainable_variables()
    # ema_op = ema.apply(tvars)

    with tf.Session(config=config) as sess:
        sess.run(init)
        # saver = tf.train.Saver(ema.variables_to_restore())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_directory))
        tf.train.write_graph(sess.graph, args.checkpoint_directory, 'graph_eval.pbtxt', as_text=True)


if __name__ == "__main__":
    main()

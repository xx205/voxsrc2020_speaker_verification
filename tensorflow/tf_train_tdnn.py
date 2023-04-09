import os
import sys
import argparse
import ast
import importlib
import multiprocessing as mp
import numpy as np

import tensorflow as tf
import horovod.tensorflow as hvd

import tf_data
import tf_scheduler
import tf_projection

def get_batch(queue, args, scp_file):
    feat_rspec_tpl = args.feat_rspec_tpl
    utt2id_pkl = args.utt2id_pkl
    cmvn_pkl = args.cmvn_pkl
    utt2kwd_pkl = args.utt2kwd_pkl
    num_classes = args.num_classes
    feat_dim = args.feat_dim
    feat_length = args.feat_length
    min_feat_length = args.min_feat_length
    max_feat_length = args.max_feat_length
    training = args.training
    specaug = args.specaug

    expansion_dim = args.expansion_dim
    batch_size = args.batch_size

    dg = tf_data.DataGenerator(scp_file, feat_rspec_tpl, utt2id_pkl, cmvn_pkl, utt2kwd_pkl, num_classes, feat_dim, feat_length, min_feat_length, max_feat_length, training, specaug)
    feature_list, label_list = [], []
    for feature, label in dg:
        feature = np.expand_dims(feature, axis=expansion_dim - 1)
        if len(feature_list) == batch_size:
            queue.put((np.stack(feature_list),
                       np.stack(label_list)))
            feature_list, label_list = [feature], [label]
        else:
            feature_list.append(feature)
            label_list.append(label)


def get_batch_synthetic(queue, args, scp_file):
    while True:
        feature = np.random.rand(args.batch_size, args.feat_length, args.feat_dim)
        feature = np.expand_dims(feature, axis=args.expansion_dim)
        label = np.random.randint(0, args.num_classes, (args.batch_size,))
        queue.put((feature, label))
        
            
def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--uuid', action='store', dest='uuid', default='********-****-****-****-************',
                        help='uuid')
    parser.add_argument('--module_source', action='store', default='tdnn_model',
                        help='module source')
    parser.add_argument('--model_id', action='store', default='tdnn',
                        help='model id')
    parser.add_argument('--expansion_dim', action='store', default=2, type=int,
                        help='expansion dim, 2 for 1D conv models like tdnn (NHWC, W=1), 3 for 2D conv models like res2net (NHWC, C=1)')
    parser.add_argument('--batch_size', action='store', default=128, type=int,
                        help='batch size')
    parser.add_argument('--dataset_length', action='store', default=-1, type=int,
                        help='dataset length')
    parser.add_argument('--num_shards_per_rank', action='store', default=1, type=int,
                        help='num shards per rank')
    parser.add_argument('--feat_rspec_tpl', action='store', default='apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:{0} ark:-|',
                        help='feature read specification template')
    parser.add_argument('--utt2id_pkl', action='store', dest='utt2id_pkl', default='None', type=str,
                        help='utt2id pkl')
    parser.add_argument('--utt2kwd_pkl', action='store', dest='utt2kwd_pkl', default='None', type=str,
                        help='utt2kwd pkl')
    parser.add_argument('--cmvn_pkl', action='store', dest='cmvn_pkl', default='None', type=str,
                        help='cmvn pkl')
    parser.add_argument('--num_classes', action='store', dest='num_classes', default=-1, type=int,
                        help='num classes')
    parser.add_argument('--projection_id', action='store', dest='projection_id', default='sc_cm_linear',
                        help='projection_id')
    parser.add_argument('--scale', action='store', dest='scale', default=32, type=float,
                        help='scale')
    # parser.add_argument('--margin', action='store', dest='margin', default=(0.15, 0.05), type=ast.literal_eval,
    parser.add_argument('--margin', action='store', dest='margin', default=0.2, type=float,
                        help='margin')
    parser.add_argument('--feat_dim', action='store', dest='feat_dim', default=-1, type=int,
                        help='feat dim')
    parser.add_argument('--feat_length', action='store', dest='feat_length', default=-1, type=int,
                        help='feat length')
    parser.add_argument('--min_feat_length', action='store', dest='min_feat_length', default=-1, type=int,
                        help='min feat length')
    parser.add_argument('--max_feat_length', action='store', dest='max_feat_length', default=-1, type=int,
                        help='max feat length')
    parser.add_argument('--training', action='store', dest='training', default=True, type=ast.literal_eval,
                        help='training')
    parser.add_argument('--use-fp16', action='store', dest='use_fp16', default=True, type=ast.literal_eval,
                        help='use fp16')
    parser.add_argument('--allreduce_post_accumulation', action='store', dest='allreduce_post_accumulation', default=True, type=ast.literal_eval,
                        help='allreduce post accumulation')
    parser.add_argument('--num_accumulation_steps', action='store', dest='num_accumulation_steps', default=1, type=int,
                        help='num accumulation steps')
    parser.add_argument('--total_epochs', action='store', dest='total_epochs', default=-1, type=int,
                        help='total epochs')
    parser.add_argument('--specaug', action='store', dest='specaug', default=False, type=ast.literal_eval,
                        help='specaug')
    parser.add_argument('--exp_dir', action='store', dest='exp_dir', default=False, type=str,
                        help='experiment directory')

    args=parser.parse_args()

    # Legacy method for enabling automatic mixed precision
    # os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
    # os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
    logger = tf.get_logger()
    logger.propagate = False
    tf.logging.set_verbosity(tf.logging.INFO)
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False

    module_source = args.module_source
    model_id = args.model_id
    expansion_dim = args.expansion_dim
    assert expansion_dim == 2 or expansion_dim == 3
    batch_size = args.batch_size
    dataset_length = args.dataset_length
    uuid = args.uuid
    num_classes = args.num_classes
    projection_id = args.projection_id

    
    model = getattr(importlib.import_module(module_source), model_id)
    projection = getattr(tf_projection, projection_id)

    NUM_EPOCHS = args.total_epochs
    EPOCH_SIZE = dataset_length // batch_size // args.num_accumulation_steps
    context = mp.get_context("spawn")

    train_queue = context.Queue(8)

    world_rank = int(os.environ.get('OMPI_COMM_WORLD_RANK',''))
    world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE',''))

    num_shards = args.num_shards
    for i in range(num_shards * world_rank, num_shards * (world_rank + 1)):
        scp_file = '{}-split/feats.{}.scp'.format(num_shards * world_size, i + 1)
        train_data_process = context.Process(target=get_batch, args=(train_queue, args, scp_file))
        # train_data_process = context.Process(target=get_batch_synthetic, args=(train_queue, args, scp_file))
        train_data_process.daemon = True
        train_data_process.start()

    # os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK','')

    hvd.init()
    if hvd.rank() == 0:
        print(args)

    EPOCH_SIZE = EPOCH_SIZE // hvd.size()

    config = tf.ConfigProto()
    # Show placement
    # config.log_device_placement=True
    # Use minimal GPU memory
    config.gpu_options.allow_growth = True
    # Enable XLA
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    # Pin GPU to be used to process local rank (one GPU per process)

    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # Forces all CPU tensors to be allocated with Cuda pinned memory
    config.gpu_options.force_gpu_compatible = True

    # Define input shape
    input_dims = [None, None, args.feat_dim]
    # For 1D conv like TDNN, input shape is N, H, 1, C
    # For 2D convlike Res2Net, input shape is N, H, W, 1
    input_dims.insert(expansion_dim, 1)

    X = tf.placeholder(tf.float32, input_dims, name="inputs")
    Y = tf.placeholder(tf.int32, [None,], name="labels")

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf_scheduler.warmup_constant_exponential_decay(0.08 / 128 * hvd.size() * batch_size * args.num_accumulation_steps, global_step, [EPOCH_SIZE * 3, EPOCH_SIZE * 13, EPOCH_SIZE * 23], EPOCH_SIZE, decay_rate=0.5, staircase=True)

    scale = args.scale
    margin = tf_scheduler.zero_linear_constant(args.margin, global_step, [EPOCH_SIZE * 3, EPOCH_SIZE * 13], EPOCH_SIZE, staircase=True)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

    # Enable automatic mixed precision
    if args.use_fp16:
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    try:
        x = model(inputs=X, training=True)
    except:
        x = model(inputs=X)

    logits = projection(x, Y, num_classes, scale=scale, margin=margin, name=projection_id)

    regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    classification_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    loss = classification_loss + regularization_loss
    accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, Y, 1), tf.float32))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        tvars = tf.trainable_variables()
        grads_and_vars = optimizer.compute_gradients(loss * 1.0 / args.num_accumulation_steps, tvars)

        # if args.num_accumulation_steps > 1:
        if True:
            local_step = tf.get_variable(name="local_step", shape=[], dtype=tf.int32, trainable=False,
                                         initializer=tf.zeros_initializer())

            accum_vars = [tf.get_variable(
                name=tvar.name.split(":")[0] + "/accum",
                shape=tvar.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer()) for tvar in tf.trainable_variables()]

            reset_step = tf.cast(tf.math.equal(local_step % args.num_accumulation_steps, 0), dtype=tf.bool)
            local_step = tf.cond(reset_step, lambda:local_step.assign(tf.ones_like(local_step)), lambda:local_step.assign_add(1))

            grads_and_vars_and_accums = [(gv[0],gv[1],accum_vars[i]) for i, gv in enumerate(grads_and_vars) if gv[0] is not None]
            grads, tvars, accum_vars = list(zip(*grads_and_vars_and_accums))

            accum_vars = tf.cond(reset_step,
                    lambda: [accum_vars[i].assign(grad) for i, grad in enumerate(grads)],
                    lambda: [accum_vars[i].assign_add(grad) for i, grad in enumerate(grads)])

            def update(accum_vars):
                if args.allreduce_post_accumulation and hvd is not None:
                    accum_vars = [hvd.allreduce(tf.convert_to_tensor(accum_var), compression=hvd.Compression.fp16 if args.use_fp16 else hvd.Compression.none) if isinstance(accum_var, tf.IndexedSlices)
                                    else hvd.allreduce(accum_var, compression=hvd.Compression.fp16 if args.use_fp16 else Compression.none) for accum_var in accum_vars]
                clipped_accum_vars, gradient_norm = tf.clip_by_global_norm(accum_vars, clip_norm=1.0)
                return optimizer.apply_gradients(list(zip(clipped_accum_vars, tvars)), global_step=global_step)

            update_step = tf.identity(tf.cast(tf.math.equal(local_step % args.num_accumulation_steps, 0), dtype=tf.bool), name="update_step")
            update_op = tf.cond(update_step, lambda: update(accum_vars), lambda: tf.no_op())
            train_op = update_op
        else:
            grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
            grads, tvars = list(zip(*grads_and_vars))
            grads = [hvd.allreduce(g, compression=hvd.Compression.fp16 if args.use_fp16 else hvd.Compression.none) for g in grads]
            clipped_grads, gradient_norm = tf.clip_by_global_norm(grads, clip_norm=1.0)
            train_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=global_step)

    tf.summary.scalar("learning_rate", learning_rate)
    # Note: these values are refreshed at every local step if num_accumulation_steps > 1
    tf.summary.scalar("classification_loss", classification_loss)
    tf.summary.scalar("regularization_loss", regularization_loss)
    tf.summary.scalar("accuracy", accuracy)

    # Note: total margin is the sum of margin and auxiliary_margin
    # For linear, total margin = 0
    # For am_linear, total margin = margin
    # For aam_linear or cm_linear, total margin = margin + 0.5 * margin * margin
    # for cm_linear_voxsrc2020, total margin = margin + 0.5 * margin
    auxiliary_margin = 0.0
    if projection_id == 'linear' or projection_id == 'am_linear' or projection_id == 'sc_am_linear':
        auxiliary_margin = 0.0
    elif projection_id == 'aam_linear' or projection_id == 'cm_linear' or projection_id == 'sc_cm_linear':
        auxiliary_margin = 0.5 * margin * margin
    elif projection_id == 'cm_linear_voxsrc2020':
        auxiliary_margin = 0.5 * margin
    else:
        raise ValueError

    tf.summary.scalar("margin", margin + auxiliary_margin)

    logging_tensors = {"global_step": global_step,
                       "classification_loss": classification_loss,
                       "regularization_loss": regularization_loss,
                       "accuracy": accuracy,
                       "learning_rate": learning_rate,
                       "margin": margin + auxiliary_margin}

    hooks = [
        hvd.BroadcastGlobalVariablesHook(0),
        # tf.train.LoggingTensorHook(tensors=logging_tensors, every_n_iter=100 * args.num_accumulation_steps),
        tf.train.LoggingTensorHook(tensors=logging_tensors, every_n_iter=1 * args.num_accumulation_steps),
        # Caution: StopAtStepHook() has different behaviors when running on CPUs and GPUs!
        tf.train.StopAtStepHook(last_step=EPOCH_SIZE * NUM_EPOCHS),
    ]

    # Initialize TensorFlow monitored training session
    checkpoint_dir = args.exp_dir if hvd.rank() == 0 else None
    scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=NUM_EPOCHS + 1))
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=checkpoint_dir,
            hooks=hooks,
            scaffold=scaffold,
            config=config,
            save_checkpoint_steps=EPOCH_SIZE) as mon_sess:
        
        while not mon_sess.should_stop():
            features, labels = train_queue.get()
            # The order of training/validation mon_sess.run() is important
            _, _step = mon_sess.run([train_op, global_step], feed_dict={X: features,
                                                                        Y: labels})

if __name__ == "__main__":
    tf.app.run()

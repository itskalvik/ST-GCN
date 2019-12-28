from model.stgcn import Model
import tensorflow as tf
from tqdm import tqdm
import argparse
import inspect
import shutil
import yaml
import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolutional Neural Network for Skeleton-Based Action Recognition')
    parser.add_argument(
        '--base-lr', type=float, default=1e-1, help='initial learning rate')
    parser.add_argument(
        '--num-classes', type=int, default=60, help='number of classes in dataset')
    parser.add_argument(
        '--batch-size', type=int, default=64, help='training batch size')
    parser.add_argument(
        '--num-epochs', type=int, default=50, help='total epochs to train')
    parser.add_argument(
        '--save-freq', type=int, default=10, help='periodicity of saving model weights')
    parser.add_argument(
        '--checkpoint-path',
        default="checkpoints/STGCN",
        help='folder to store model weights')
    parser.add_argument(
        '--log-dir',
        default="logs/STGCN",
        help='folder to store model-definition/training-logs/hyperparameters')
    parser.add_argument(
        '--train-data-path',
        default="data/ntu/xview/train_data",
        help='path to folder with training dataset tfrecord files')
    parser.add_argument(
        '--test-data-path',
        default="data/ntu/xview/val_data",
        help='path to folder with testing dataset tfrecord files')
    parser.add_argument(
        '--steps',
        type=int,
        default=[10, 40],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate, eg: 10 50')
    parser.add_argument(
        '--gpus',
        default=None,
        nargs='+',
        help='list of gpus to use for training, eg: "/gpu:0" "/gpu:1"')

    return parser


def save_arg(arg):
    # save arg
    arg_dict = vars(arg)
    if not os.path.exists(arg.log_dir):
        os.makedirs(arg.log_dir)
    with open(os.path.join(arg.log_dir, "config.yaml"), 'w') as f:
        yaml.dump(arg_dict, f)


'''
get_dataset: Returns a tensorflow dataset object with features and one hot
encoded label data
Args:
  directory       : Path to folder with TFRecord files for dataset
  num_classes     : Number of classes in dataset for one hot encoding
  batch_size      : Represents the number of consecutive elements of this
                    dataset to combine in a single batch.
  drop_remainder  : If True, the last batch will be dropped in the case it has
                    fewer than batch_size elements. Defaults to False
  shuffle         : If True, the data samples will be shuffled randomly.
                    Defaults to False
  shuffle_size    : Size of buffer used to hold data for shuffling
Returns:
  The Dataset with features and one hot encoded label data
'''
def get_dataset(directory, num_classes=60, batch_size=32, drop_remainder=False,
                shuffle=False, shuffle_size=1000):
    # dictionary describing the features.
    feature_description = {
        'features': tf.io.FixedLenFeature([], tf.string),
        'label'     : tf.io.FixedLenFeature([], tf.int64)
    }

    # parse each proto and, the features within
    def _parse_feature_function(example_proto):
        features = tf.io.parse_single_example(example_proto, feature_description)
        data =  tf.io.parse_tensor(features['features'], tf.float32)
        label = tf.one_hot(features['label'], num_classes)
        data = tf.reshape(data, (3, 300, 25, 2))
        return data, label

    records = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith("tfrecord")]
    dataset = tf.data.TFRecordDataset(records, num_parallel_reads=len(records))
    dataset = dataset.map(_parse_feature_function)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(batch_size)
    if shuffle:
        dataset = dataset.shuffle(shuffle_size)
    return dataset


'''
test_step: gets model prediction for given samples
Args:
  features: tensor with features
'''
@tf.function
def test_step(features):
    logits = model(features, training=False)
    return tf.nn.softmax(logits)


'''
train_step: trains model with cross entropy loss
Args:
  features    : tensor with features
  labels      : one hot encoded labels
'''
@tf.function
def train_step(features, labels):
  def step_fn(features, labels):
    with tf.GradientTape() as tape:
      logits = model(features, training=True)
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=labels)
      loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
    train_acc(labels, logits)
    train_acc_top_5(labels, logits)
    cross_entropy_loss(loss)
  strategy.experimental_run_v2(step_fn, args=(features, labels,))


if __name__ == "__main__":
    parser = get_parser()
    arg = parser.parse_args()

    base_lr         = arg.base_lr
    num_classes     = arg.num_classes
    epochs          = arg.num_epochs
    checkpoint_path = arg.checkpoint_path
    log_dir         = arg.log_dir
    train_data_path = arg.train_data_path
    test_data_path  = arg.test_data_path
    save_freq       = arg.save_freq
    steps           = arg.steps
    batch_size      = arg.batch_size
    gpus            = arg.gpus
    strategy        = tf.distribute.MirroredStrategy(arg.gpus)
    global_batch_size = arg.batch_size*strategy.num_replicas_in_sync
    arg.gpus        = strategy.num_replicas_in_sync

    #copy hyperparameters and model definition to log folder
    save_arg(arg)
    shutil.copy2(inspect.getfile(Model), arg.log_dir)

    '''
    Get tf.dataset objects for training and testing data
    Data shape: features - batch_size, 3, 300, 25, 2
                labels   - batch_size, num_classes
    '''
    train_data = get_dataset(train_data_path,
                             num_classes=num_classes,
                             batch_size=global_batch_size,
                             drop_remainder=True,
                             shuffle=True)
    train_data = strategy.experimental_distribute_dataset(train_data)

    test_data = get_dataset(test_data_path,
                            num_classes=num_classes,
                            batch_size=batch_size,
                            drop_remainder=False,
                            shuffle=False)

    boundaries = [(step*40000)//batch_size for step in steps]
    values = [base_lr]*(len(steps)+1)
    for i in range(1, len(steps)+1):
        values[i] *= 0.1**i
    learning_rate  = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    with strategy.scope():
        model        = Model(num_classes=num_classes)
        optimizer    = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                               momentum=0.9,
                                               nesterov=True)
        ckpt         = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                  checkpoint_path,
                                                  max_to_keep=5)

        # keras metrics to hold accuracies and loss
        cross_entropy_loss   = tf.keras.metrics.Mean(name='cross_entropy_loss')
        train_acc            = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
        train_acc_top_5      = tf.keras.metrics.TopKCategoricalAccuracy(name='train_acc_top_5')

    epoch_test_acc       = tf.keras.metrics.CategoricalAccuracy(name='epoch_test_acc')
    epoch_test_acc_top_5 = tf.keras.metrics.TopKCategoricalAccuracy(name='epoch_test_acc_top_5')
    test_acc_top_5       = tf.keras.metrics.TopKCategoricalAccuracy(name='test_acc_top_5')
    test_acc             = tf.keras.metrics.CategoricalAccuracy(name='test_acc')
    summary_writer       = tf.summary.create_file_writer(log_dir)

    # Get 1 batch from train dataset to get graph trace of train and test functions
    for data in test_data:
        features, labels = data
        break

    # add graph of train and test functions to tensorboard graphs
    # Note:
    # graph training is True on purpose, allows tensorflow to get all the
    # variables, which is required for the first call of @tf.function function
    tf.summary.trace_on(graph=True)
    train_step(features, labels)
    with summary_writer.as_default():
      tf.summary.trace_export(name="training_trace",step=0)
    tf.summary.trace_off()

    tf.summary.trace_on(graph=True)
    test_step(features)
    with summary_writer.as_default():
      tf.summary.trace_export(name="testing_trace", step=0)
    tf.summary.trace_off()

    # start training
    train_iter = 0
    test_iter = 0
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch+1))
        print("Training: ")
        with strategy.scope():
            for features, labels in tqdm(train_data):
                train_step(features, labels)
                with summary_writer.as_default():
                    tf.summary.scalar("cross_entropy_loss",
                                      cross_entropy_loss.result(),
                                      step=train_iter)
                    tf.summary.scalar("train_acc",
                                      train_acc.result(),
                                      step=train_iter)
                    tf.summary.scalar("train_acc_top_5",
                                      train_acc_top_5.result(),
                                      step=train_iter)
                cross_entropy_loss.reset_states()
                train_acc.reset_states()
                train_acc_top_5.reset_states()
                train_iter += 1

        print("Testing: ")
        for features, labels in tqdm(test_data):
            y_pred = test_step(features)
            test_acc(labels, y_pred)
            epoch_test_acc(labels, y_pred)
            test_acc_top_5(labels, y_pred)
            epoch_test_acc_top_5(labels, y_pred)
            with summary_writer.as_default():
                tf.summary.scalar("test_acc",
                                  test_acc.result(),
                                  step=test_iter)
                tf.summary.scalar("test_acc_top_5",
                                  test_acc_top_5.result(),
                                  step=test_iter)
            test_acc.reset_states()
            test_acc_top_5.reset_states()
            test_iter += 1
        with summary_writer.as_default():
            tf.summary.scalar("epoch_test_acc",
                              epoch_test_acc.result(),
                              step=epoch)
            tf.summary.scalar("epoch_test_acc_top_5",
                              epoch_test_acc_top_5.result(),
                              step=epoch)
        epoch_test_acc.reset_states()
        epoch_test_acc_top_5.reset_states()

        if (epoch + 1) % save_freq == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                ckpt_save_path))

    ckpt_save_path = ckpt_manager.save()
    print('Saving final checkpoint for epoch {} at {}'.format(epochs,
                                                              ckpt_save_path))

from model.agcn import AGCN
import tensorflow as tf
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
        description='Two-Stream Adaptive Graph Convolutional Neural Network for Skeleton-Based Action Recognition')
    parser.add_argument(
        '--num-classes', type=int, default=60, help='number of classes in dataset')
    parser.add_argument(
        '--batch-size', type=int, default=16, help='training batch size')
    parser.add_argument(
        '--joint_checkpoint_path',
        default="checkpoints/xview-joint/",
        help='folder to store model weights')
    parser.add_argument(
        '--bone_checkpoint_path',
        default="checkpoints/xview-bone/",
        help='folder to store model weights')
    parser.add_argument(
        '--log-dir',
        default="logs/2s-AGCN-xview",
        help='folder to store model-definition/training-logs/hyperparameters')
    parser.add_argument(
        '--joint_test_data_path',
        default="data/ntu/xview/val_data_joint",
        help='path to folder with testing dataset tfrecord files')
    parser.add_argument(
        '--bone_test_data_path',
        default="data/ntu/xview/val_data_bone",
        help='path to folder with testing dataset tfrecord files')
    return parser

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
                shuffle=False, shuffle_size=0):
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
  joint_features: tensor with features
  bone_features : tensor with features
'''
@tf.function
def test_step(joint_features, bone_features):
    joint_logits = joint_model(joint_features, training=False)
    bone_logits  = bone_model(bone_features, training=False)
    return tf.nn.softmax(joint_logits) + tf.nn.softmax(bone_logits)


if __name__ == "__main__":
    parser = get_parser()
    arg = parser.parse_args()

    num_classes           = arg.num_classes
    joint_checkpoint_path = arg.joint_checkpoint_path
    bone_checkpoint_path  = arg.bone_checkpoint_path
    joint_test_data_path  = arg.joint_test_data_path
    bone_test_data_path   = arg.bone_test_data_path
    batch_size            = arg.batch_size
    log_dir               = arg.log_dir

    '''
    Get tf.dataset objects for training and testing data
    Data shape: features - batch_size, 3, 300, 25, 2
                labels   - batch_size, num_classes
    '''
    bone_test_data = get_dataset(bone_test_data_path,
                                 num_classes=num_classes,
                                 batch_size=batch_size,
                                 drop_remainder=False,
                                 shuffle=False)
    joint_test_data = get_dataset(joint_test_data_path,
                                  num_classes=num_classes,
                                  batch_size=batch_size,
                                  drop_remainder=False,
                                  shuffle=False)

    # Load bone model
    bone_model        = AGCN(num_classes=num_classes)
    bone_ckpt         = tf.train.Checkpoint(model=bone_model)
    bone_ckpt_manager = tf.train.CheckpointManager(bone_ckpt,
                                              bone_checkpoint_path,
                                              max_to_keep=5)
    bone_ckpt.restore(bone_ckpt_manager.latest_checkpoint)

    # Load joint model
    joint_model        = AGCN(num_classes=num_classes)
    joint_ckpt         = tf.train.Checkpoint(model=joint_model)
    joint_ckpt_manager = tf.train.CheckpointManager(joint_ckpt,
                                                    joint_checkpoint_path,
                                                    max_to_keep=5)
    joint_ckpt.restore(joint_ckpt_manager.latest_checkpoint)

    epoch_test_acc       = tf.keras.metrics.CategoricalAccuracy(name='epoch_test_acc')
    epoch_test_acc_top_5 = tf.keras.metrics.TopKCategoricalAccuracy(name='epoch_test_acc_top_5')
    test_acc_top_5       = tf.keras.metrics.TopKCategoricalAccuracy(name='test_acc_top_5')
    test_acc             = tf.keras.metrics.CategoricalAccuracy(name='test_acc')
    summary_writer       = tf.summary.create_file_writer(log_dir)

    print("Testing: ")
    test_iter = 0
    for (joint_features, labels), (bone_features, _) in zip(joint_test_data, bone_test_data):
        y_pred = test_step(joint_features, bone_features)
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
                          step=0)
        tf.summary.scalar("epoch_test_acc_top_5",
                          epoch_test_acc_top_5.result(),
                          step=0)
    epoch_test_acc.reset_states()
    epoch_test_acc_top_5.reset_states()

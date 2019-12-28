import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(features, label):
    feature = {
        'features' : _bytes_feature(tf.io.serialize_tensor(features.astype(np.float32))),
        'label'     : _int64_feature(label)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def gen_tfrecord_data(num_shards, label_path, data_path, dest_folder, shuffle):
    label_path = Path(label_path)
    if not (label_path.exists()):
        print('Label file does not exist')
        return

    data_path = Path(data_path)
    if not (data_path.exists()):
        print('Data file does not exist')
        return

    try:
        with open(label_path) as f:
            _, labels = pickle.load(f)
    except:
        # for pickle file from python2
        with open(label_path, 'rb') as f:
            _, labels = pickle.load(f, encoding='latin1')

    # Datashape: Total_samples, 3, 300, 25, 2
    data   = np.load(data_path, mmap_mode='r')
    labels = np.array(labels)

    if len(labels) != len(data):
        print("Data and label lengths didn't match!")
        print("Data size: {} | Label Size: {}".format(data.shape, labels.shape))
        return -1

    print("Data shape:", data.shape)
    if shuffle:
        p = np.random.permutation(len(labels))
        labels = labels[p]
        data = data[p]

    dest_folder = Path(dest_folder)
    if not (dest_folder.exists()):
        os.mkdir(dest_folder)

    step = len(labels)//num_shards
    for shard in tqdm(range(num_shards)):
        tfrecord_data_path = os.path.join(dest_folder, data_path.name.split(".")[0]+"-"+str(shard)+".tfrecord")
        with tf.io.TFRecordWriter(tfrecord_data_path) as writer:
            for i in range(shard*step, (shard*step)+step if shard < num_shards-1 else len(labels)):
                writer.write(serialize_example(data[i], labels[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data TFRecord Converter')
    parser.add_argument('--num-shards',
                        type=int,
                        default=40,
                        help='number of files to split dataset into')
    parser.add_argument('--label-path',
                        required=True,
                        help='path to pkl file with labels')
    parser.add_argument('--shuffle',
                        required=True,
                        help='setting it to True will shuffle the labels and data together')
    parser.add_argument('--data-path',
                        required=True,
                        help='path to npy file with data')
    parser.add_argument('--dest-folder',
                        required=True,
                        help='path to folder in which tfrecords will be stored')
    arg = parser.parse_args()

    gen_tfrecord_data(arg.num_shards,
                      arg.label_path,
                      arg.data_path,
                      arg.dest_folder,
                      arg.shuffle)

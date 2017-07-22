# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from PIL import Image

if __name__ == '__main__':
    main()  


def main():
    load_mnist('train.csv')


def load_mnist(mn_dir):
    return read_data_sets(mn_dir)


def read_data_sets(mn_dir):
    _ds = pd.read_csv(mn_dir)
    images = np.asarray(_ds.iloc[:, 1:].values) 
    labels = np.asarray(_ds.iloc[:, 0].values).reshape(-1, 1)
    if len(_ds.columns) == 785:
        images = np.asarray(_ds.iloc[:, 1:].values)
        labels = np.asarray(_ds.iloc[:, 0].values).reshape(-1, 1)
    else:
        images = np.asarray(_ds.iloc[:, :].values).astype(np.float32)
        labels = np.random.randint(0, 10, (images.shape[0], 1))
    return dataset(images, labels)


class dataset(object):
    """docstring for dataset"""

    def __init__(self,
                 images,
                 labels,
                 reshape=False,
                 shape_image=(28, 28, 1),
                 one_hot=True,
                 num_labels=10
                 ):
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        if reshape:
            images = images.reshape((-1,) + shape_image)
        if one_hot:
            labels = self.dense_to_one_hot(labels, num_labels)
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def dense_to_one_hot(self, labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=False):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
                # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

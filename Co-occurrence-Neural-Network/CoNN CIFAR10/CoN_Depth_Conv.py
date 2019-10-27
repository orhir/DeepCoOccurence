# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed u co_shape
#         co_shapeself.w_shape = nder the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python import debug as tf_debug
import cifar10_input

import json
from datetime import datetime
import time

import tensorflow as tf
from tensorflow.python.client import timeline
import CoL_layer
import cifar10
from tensorflow import keras
import datetime as dt
import tkinter
import matplotlib.pyplot as plt
from DepthwiseConv3D import DepthwiseConv3D

import psutil
import gc

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 0,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.


class CoLL(keras.Model):
    def __init__(self, co_shape, w_shape, co_initializer=None, w_initializer=None, strides=1, cname="col_"):
        super().__init__(name='conn_layer')
        if co_initializer is None:
            co_initializer = tf.initializers.he_normal()
            # co_initializer = tf.truncated_normal_initializer(stddev=0.5, dtype=tf.float32)
        if w_initializer is None:
            w_initializer = tf.initializers.he_normal()
            # w_initializer = tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32)

        self.cname = cname
        self.co_matrix = CoL_layer.co_variable_initializer(co_initializer, co_shape, cname + 'co_matrix')
        self.w_spatial = CoL_layer.w_variable_initializer(w_initializer, w_shape, cname + 'w_filter')
        self.co_shape = co_shape
        self.w_shape = w_shape
        self.strides = strides
        self.num_bins = co_shape[0]
        self.dc3d = DepthwiseConv3D(kernel_size=(3, 3, 3), groups=self.num_bins, padding='SAME',
                                    depthwise_initializer='he_normal')


    def call(self, x):


        # the size parameters
        x_shape = tf.shape(x)

        # quantization step of the input for index decision
        input_index = CoL_layer.quantization_input_as_bins(x, self.num_bins)

        vec_input_index = tf.reshape(input_index, [-1, 1])

        # the variable with final result
        sp0, sp1, sp2, sp3 = x.get_shape().as_list()
        sp1 = int(sp1 / self.strides)
        total_conv = tf.zeros([x_shape[0], sp1, sp1, sp3])  # Why is dim2 == dim1? can it be sp2? Todo:

        temp_strides = tf.zeros([self.strides])
        # shape of input data
        x_shape = tf.shape(x)
        w_shape = tf.shape(self.w_spatial)
        cof_matrix_bin = tf.reshape(tf.gather_nd(self.co_matrix[0, :], vec_input_index), x_shape)
        cof_matrix_bin = tf.expand_dims(cof_matrix_bin, axis=4)
        temp_mask = tf.cast(tf.equal(input_index, 0), tf.float32)
        temp_mask = tf.expand_dims(temp_mask, axis=4)

        for i in range(1, self.num_bins):
            temp_cof_matrix_bin = tf.reshape(tf.gather_nd(self.co_matrix[i, :], vec_input_index), x_shape)
            temp_cof_matrix_bin = tf.expand_dims(temp_cof_matrix_bin, axis=4)
            cof_matrix_bin = tf.concat([cof_matrix_bin, temp_cof_matrix_bin], axis=4)
            cur_temp_mask = tf.cast(tf.equal(input_index, i), tf.float32)
            cur_temp_mask = tf.expand_dims(cur_temp_mask, axis=4)
            temp_mask = tf.concat([temp_mask, cur_temp_mask], axis=4)

        temp_input = tf.multiply(cof_matrix_bin, tf.expand_dims(x, axis=4))

        # w_3d = tf.reshape(self.w_spatial, [w_shape[0], w_shape[1], w_shape[2], 1, 1])
        # temp_strides = tf.zeros([self.strides])
        # strid = temp_strides.get_shape().as_list()

        # dc3d.filters = w_3d
        temp_result = self.dc3d(temp_input)

        temp_result = tf.multiply(temp_result, temp_mask)
        total_conv = tf.math.reduce_sum(temp_result, axis=4)
        return total_conv

        # while loop of bins in co-occurrence matrix
        # condition_per_bins = lambda i, *_: i < self.num_bins
        # _, total_conv, self.w_spatial, self.co_matrix, _, _, _, _ = \
        #     tf.while_loop(condition_per_bins, CoL_layer.body_per_bins,
        #                                                 [0, total_conv, self.w_spatial, self.co_matrix, x, input_index,
        #                                                  vec_input_index, temp_strides]) #TODO: check returned vals
        #
        # conn = total_conv
        # return conn


class CIFAR10Model(keras.Model):
    def __init__(self):
        super(CIFAR10Model, self).__init__(name='cifar_cnn')
        self.conv1 = keras.layers.Conv2D(64, 5,
                                         padding='same',
                                         activation=tf.nn.relu,
                                         # kernel_initializer=tf.initializers.variance_scaling,
                                         # kernel_regularizer=keras.regularizers.l2(l=0.0)
                                         )
        self.max_pool2d = keras.layers.MaxPooling2D((3, 3), (2, 2), padding='same')
        # self.max_norm = keras.layers.BatchNormalization()

        self.CoL = CoLL(co_shape=[5, 5],
                        w_shape=[3, 3, 3],
                        strides=1,
                        cname='conn2')

        # self.conv2 = keras.layers.Conv2D(64, 5,
        #                                  padding='same',
        #                                  activation=tf.nn.relu,
        #                                  kernel_initializer=tf.initializers.variance_scaling,
        #                                  kernel_regularizer=keras.regularizers.l2(l=0.001))

        self.flatten = keras.layers.Flatten()
        # self.dense = keras.layers.Dense(64, activation=tf.nn.relu,
        #                                 bias_initializer=tf.constant_initializer(0.1)
        #                               # kernel_initializer=tf.initializers.variance_scaling,
        #                               # kernel_regularizer=keras.regularizers.l2(l=0.004)
        #                                 )

        # w = tf.Variable([64 * [1.0]])

        self.fc1 = keras.layers.Dense(192, activation=tf.nn.relu,
                                      # bias_initializer=tf.constant_initializer(0.1),
                                      # kernel_initializer=tf.initializers.variance_scaling,
                                      kernel_regularizer=keras.regularizers.l2(l=0.004))
        self.fc2 = keras.layers.Dense(10, activation=tf.nn.relu,
                                      # bias_initializer=tf.constant_initializer(0.1),
                                      # kernel_initializer=tf.initializers.variance_scaling,
                                      kernel_regularizer=keras.regularizers.l2(l=0.004))

        # self.dropout = keras.layers.Dropout(0.5)
        # self.fc2 = keras.layers.Dense(10)
        self.softmax = keras.layers.Softmax()
        self.CoLBias = tf.Variable([0.1] * 64, name='CoL_bias')

    def call(self, x):
        x = self.max_pool2d(self.conv1(x))
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        x = tf.nn.bias_add(self.CoL(x), self.CoLBias)
        # x = self.CoL(x)
        x = tf.nn.relu(x, name="CoL")
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        x = self.max_pool2d(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train():
    """Train CIFAR-10 for a number of steps."""
    tfe = tf.contrib.eager
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    num_samples = len(x_train)
    # num_samples = 200
    x_train, y_train = x_train[:num_samples], y_train[:num_samples]
    batch_size = 8
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    global_step = tf.train.get_or_create_global_step()

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # TODO: Fix Shuffler
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(tf.cast(batch_size, tf.int64)).shuffle(
        num_samples)
    train_dataset = train_dataset.map(
        lambda x, y: (tf.div(tf.cast(x, tf.float32), 255.0), tf.reshape(tf.one_hot(y, 10), (-1, 10))))
    train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
    train_dataset = train_dataset.repeat()

    valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(tf.cast(batch_size, tf.int64)).shuffle(
        10000)
    valid_dataset = valid_dataset.map(
        lambda x, y: (tf.div(tf.cast(x, tf.float32), 255.0), tf.reshape(tf.one_hot(y, 10), (-1, 10))))
    valid_dataset = valid_dataset.repeat()

    model = CIFAR10Model()
    tf.get_default_graph().finalize()

    # keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    loss_history = []
    mem = []

    for i in range(1):
        for (itern, (x, y)) in enumerate(train_dataset):
            gc.collect()
            tf.keras.backend.clear_session()
            # plt.imshow(x[40].numpy())
            # plt.show()
            # for itern, (x, y) in enumerate(train_dataset):


            start_time = time.time()
            with tfe.GradientTape() as tape:
                predicted = model(x)
                curr_loss = tf.losses.softmax_cross_entropy(y, predicted)
                grads = tape.gradient(curr_loss, model.variables)
                optimizer.apply_gradients(zip(grads, model.variables),
                                          global_step=tf.train.get_or_create_global_step())

            if itern % FLAGS.log_frequency == 0:
                mem_usage = psutil.virtual_memory().used / 2 ** 30
                current_time = time.time()
                duration = current_time - start_time
                sec_per_batch = float(duration / FLAGS.log_frequency)
                examples_per_sec = FLAGS.log_frequency * batch_size / duration
                format_str = ('%s: epoch %d, step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), i, itern, curr_loss,
                                    examples_per_sec, sec_per_batch))
                loss_history.append(curr_loss.numpy())
                mem.append(mem_usage)

    plt.plot(mem)
    # plt.plot(loss_history)
    plt.xlabel('Batch #')
    plt.ylabel('Loss [entropy]')
    plt.show()


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    tf.enable_eager_execution()
    tf.app.run()

#     TODO: Copy Ira's network
#     TODO: Build a training loop

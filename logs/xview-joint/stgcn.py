from graph.ntu_rgb_d import Graph
import tensorflow as tf
import numpy as np

REGULARIZER = tf.keras.regularizers.l2(l=0.0001)
INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.,
                                                    mode="fan_out",
                                                    distribution="truncated_normal")


"""The basic module for applying a spatial graph convolution.
    Args:
        filters (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, C, T, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size
            :math:`K` is the spatial kernel size
            :math:`T` is a length of the sequence
            :math:`V` is the number of graph nodes
            :math:`C` is the number of incoming channels
"""
class SGCN(tf.keras.Model):
    def __init__(self, filters, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv2D(filters*kernel_size,
                                           kernel_size=1,
                                           padding='same',
                                           kernel_initializer=INITIALIZER,
                                           data_format='channels_first',
                                           kernel_regularizer=REGULARIZER)

    # N, C, T, V
    def call(self, x, A, training):
        x = self.conv(x)

        N = tf.shape(x)[0]
        C = tf.shape(x)[1]
        T = tf.shape(x)[2]
        V = tf.shape(x)[3]

        x = tf.reshape(x, [N, self.kernel_size, C//self.kernel_size, T, V])
        x = tf.einsum('nkctv,kvw->nctw', x, A)
        return x, A


"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        filters (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        activation (activation function/name, optional): activation function to use
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
        downsample (bool, optional): If ``True``, applies a downsampling residual mechanism. Default: ``True``
                                     the value is used only when residual is ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
"""
class STGCN(tf.keras.Model):
    def __init__(self, filters, kernel_size=[9, 3], stride=1, activation='relu',
                 residual=True, downsample=False):
        super().__init__()
        self.sgcn = SGCN(filters, kernel_size=kernel_size[1])

        self.tgcn = tf.keras.Sequential()
        self.tgcn.add(tf.keras.layers.BatchNormalization(axis=1))
        self.tgcn.add(tf.keras.layers.Activation(activation))
        self.tgcn.add(tf.keras.layers.Conv2D(filters,
                                                kernel_size=[kernel_size[0], 1],
                                                strides=[stride, 1],
                                                padding='same',
                                                kernel_initializer=INITIALIZER,
                                                data_format='channels_first',
                                                kernel_regularizer=REGULARIZER))
        self.tgcn.add(tf.keras.layers.BatchNormalization(axis=1))

        self.act = tf.keras.layers.Activation(activation)

        if not residual:
            self.residual = lambda x, training=False: 0
        elif residual and stride == 1 and not downsample:
            self.residual = lambda x, training=False: x
        else:
            self.residual = tf.keras.Sequential()
            self.residual.add(tf.keras.layers.Conv2D(filters,
                                                        kernel_size=[1, 1],
                                                        strides=[stride, 1],
                                                        padding='same',
                                                        kernel_initializer=INITIALIZER,
                                                        data_format='channels_first',
                                                        kernel_regularizer=REGULARIZER))
            self.residual.add(tf.keras.layers.BatchNormalization(axis=1))

    def call(self, x, A, training):
        res = self.residual(x, training=training)
        x, A = self.sgcn(x, A, training=training)
        x = self.tgcn(x, training=training)
        x += res
        x = self.act(x)
        return x, A


"""Spatial temporal graph convolutional networks.
    Args:
        num_class (int): Number of classes for the classification task
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
"""
class Model(tf.keras.Model):
    def __init__(self, num_classes=60):
        super().__init__()

        graph = Graph()
        self.A = tf.Variable(graph.A,
                             dtype=tf.float32,
                             trainable=False,
                             name='adjacency_matrix')

        self.data_bn = tf.keras.layers.BatchNormalization(axis=1)

        self.STGCN_layers = []
        self.STGCN_layers.append(STGCN(64, residual=False))
        self.STGCN_layers.append(STGCN(64))
        self.STGCN_layers.append(STGCN(64))
        self.STGCN_layers.append(STGCN(64))
        self.STGCN_layers.append(STGCN(128, stride=2, downsample=True))
        self.STGCN_layers.append(STGCN(128))
        self.STGCN_layers.append(STGCN(128))
        self.STGCN_layers.append(STGCN(256, stride=2, downsample=True))
        self.STGCN_layers.append(STGCN(256))
        self.STGCN_layers.append(STGCN(256))

        self.pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')

        self.logits = tf.keras.layers.Conv2D(num_classes,
                                             kernel_size=1,
                                             padding='same',
                                             kernel_initializer=INITIALIZER,
                                             data_format='channels_first',
                                             kernel_regularizer=REGULARIZER)

    def call(self, x, training):
        N = tf.shape(x)[0]
        C = tf.shape(x)[1]
        T = tf.shape(x)[2]
        V = tf.shape(x)[3]
        M = tf.shape(x)[4]

        x = tf.transpose(x, perm=[0, 4, 3, 1, 2])
        x = tf.reshape(x, [N * M, V * C, T])
        x = self.data_bn(x, training=training)
        x = tf.reshape(x, [N, M, V, C, T])
        x = tf.transpose(x, perm=[0, 1, 3, 4, 2])
        x = tf.reshape(x, [N * M, C, T, V])

        A = self.A
        for layer in self.STGCN_layers:
            x, A = layer(x, A, training=training)

        # N*M,C,T,V
        x = self.pool(x)
        x = tf.reshape(x, [N, M, -1, 1, 1])
        x = tf.reduce_mean(x, axis=1)
        x = self.logits(x)
        x = tf.reshape(x, [N, -1])

        return x

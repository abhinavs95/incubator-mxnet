# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# This example is inspired by https://github.com/jason71995/Keras-GAN-Library,
# https://github.com/kazizzad/DCGAN-Gluon-MxNet/blob/master/MxnetDCGAN.ipynb
# https://github.com/apache/incubator-mxnet/blob/master/example/gluon/dc_gan/dcgan.py

import mxnet as mx
from mxnet import nd, sym
from mxnet import gluon, autograd
from mxnet.gluon import Block, nn
from mxnet.gluon.block import HybridBlock


EPSILON = 1e-08
POWER_ITERATION = 1

class SNConv2D(HybridBlock):
    """ Customized Conv2D to feed the conv with the weight that we apply spectral normalization """

    def __init__(self, num_filter, kernel_size,
                 strides, padding, in_channels,
                 ctx=mx.cpu(), iterations=1):

        super(SNConv2D, self).__init__()

        self.num_filter = num_filter
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.in_channels = in_channels
        self.iterations = iterations
        self.ctx = ctx

        with self.name_scope():
            self.u = self.params.get_constant('u', value=mx.ndarray.random.normal(shape=(1, num_filter)))

        self.output = nn.Conv2D(num_filter, kernel_size, strides, padding, in_channels=in_channels, use_bias=False)


    def _spectral_norm(self):
        """ spectral normalization """
        w = self.output.weight.var()
        w_mat = sym.reshape(w, [self.num_filter, -1])

        _v = None

        for _ in range(POWER_ITERATION):
            _v = sym.L2Normalization(sym.dot(self.u.var(), w_mat))
            self.u = sym.L2Normalization(sym.dot(_v, sym.transpose(w_mat)))

        sigma = sym.sum(sym.dot(self.u.var(), w_mat) * _v)
        sigma = sym.where(sigma.__eq__(0.), EPSILON * sym.ones(1), sigma)

        w = w / sigma
        self.output.weight = w
        
        return
# weight not updating???

    def hybrid_forward(self, F, x, u):#, weight):
        # x shape is batch_size x in_channels x height x width
        self._spectral_norm
        return self.output(x)#, weight=self._spectral_norm)



def get_generator():
    """ construct and return generator """
    g_net = gluon.nn.HybridSequential()
    with g_net.name_scope():

        g_net.add(gluon.nn.Conv2DTranspose(
            channels=512, kernel_size=4, strides=1, padding=0, use_bias=False))
        g_net.add(gluon.nn.BatchNorm())
        g_net.add(gluon.nn.LeakyReLU(0.2))

        g_net.add(gluon.nn.Conv2DTranspose(
            channels=256, kernel_size=4, strides=2, padding=1, use_bias=False))
        g_net.add(gluon.nn.BatchNorm())
        g_net.add(gluon.nn.LeakyReLU(0.2))

        g_net.add(gluon.nn.Conv2DTranspose(
            channels=128, kernel_size=4, strides=2, padding=1, use_bias=False))
        g_net.add(gluon.nn.BatchNorm())
        g_net.add(gluon.nn.LeakyReLU(0.2))

        g_net.add(gluon.nn.Conv2DTranspose(
            channels=64, kernel_size=4, strides=2, padding=1, use_bias=False))
        g_net.add(gluon.nn.BatchNorm())
        g_net.add(gluon.nn.LeakyReLU(0.2))

        g_net.add(gluon.nn.Conv2DTranspose(channels=3, kernel_size=4, strides=2, padding=1, use_bias=False))
        g_net.add(gluon.nn.Activation('tanh'))

    return g_net


def get_descriptor(ctx):
    """ construct and return descriptor """
    d_net = gluon.nn.HybridSequential()
    with d_net.name_scope():

        d_net.add(SNConv2D(num_filter=64, kernel_size=4, strides=2, padding=1, in_channels=3, ctx=ctx))
        d_net.add(gluon.nn.LeakyReLU(0.2))

        d_net.add(SNConv2D(num_filter=128, kernel_size=4, strides=2, padding=1, in_channels=64, ctx=ctx))
        d_net.add(gluon.nn.LeakyReLU(0.2))

        d_net.add(SNConv2D(num_filter=256, kernel_size=4, strides=2, padding=1, in_channels=128, ctx=ctx))
        d_net.add(gluon.nn.LeakyReLU(0.2))

        d_net.add(SNConv2D(num_filter=512, kernel_size=4, strides=2, padding=1, in_channels=256, ctx=ctx))
        d_net.add(gluon.nn.LeakyReLU(0.2))

        d_net.add(SNConv2D(num_filter=1, kernel_size=4, strides=1, padding=0, in_channels=512, ctx=ctx))

    return d_net

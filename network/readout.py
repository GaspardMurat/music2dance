# -*- coding: utf-8 -*-
"""Convolutional-recurrent layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers.recurrent import _generate_dropout_mask
from keras.layers.recurrent import _standardize_args

import numpy as np
import warnings
from keras.engine.base_layer import InputSpec, Layer
from keras.utils import conv_utils
from keras.layers.recurrent import RNN
from keras.utils.generic_utils import has_arg
from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import transpose_shape


class ConvRNN2D_readout(RNN):
    """Base class for convolutional-recurrent layers.

    # Arguments
        cell: A RNN cell instance. A RNN cell is a class that has:
            - a `call(input_at_t, states_at_t)` method, returning
              `(output_at_t, states_at_t_plus_1)`.
            - a `state_size` attribute. This can be a single integer (single state)
              in which case it is the number of channels of the recurrent state
              (which should be the same as the number of channels of the cell
              output). This can also be a list/tuple of integers
              (one size per state). In this case, the first entry (`state_size[0]`)
              should be the same as the size of the cell output.
        return_sequences: Boolean. Whether to return the last output.
            in the output sequence, or the full sequence.
        input_shape: Use this argument to specify the shape of the
            input when this layer is the first one in a model.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        - if `return_sequences`: 5D tensor with shape:
            `(samples, timesteps,
              filters, new_rows, new_cols)` if data**kwargs_format='channels_first'
            or 5D tensor with shape:
            `(samples, timesteps,
              new_rows, new_cols, filters)` if data_format='channels_last'.
        - else, 4D tensor with shape:
            `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.

    # Note on specifying the initial state of RNNs
        You can specify the initial state of RNN layers symbolically by
        calling them with the keyword argument `initial_state`. The value of
        `initial_state` should be a tensor or list of tensors representing
        the initial state of the RNN layer.

        You can specify the initial state of RNN layers numerically by
        calling `reset_states` with the keyword argument `states`. The value of
        `states` should be a numpy array or list of numpy arrays representing
        the initial state of the RNN layer.
    """

    def __init__(self, cell,
                 timestep_out,
                 return_sequences=False,
                 initial_state=None,
                 **kwargs):

        if not hasattr(cell, 'call'):
            raise ValueError('`cell` should have a `call` method. '
                             'The ConvRNN was passed:', cell)
        if not hasattr(cell, 'state_size'):
            raise ValueError('The RNN cell should have '
                             'an attribute `state_size` '
                             '(tuple of array, '
                             'one array per RNN state).')

        super(ConvRNN2D_readout, self).__init__(cell,
                                                timestep_out,
                                                return_sequences,
                                                initial_state,
                                                **kwargs)

        self.return_sequences = return_sequences
        self.return_state = False
        self.go_backwards = False
        self.stateful = False
        self.unroll = False

        self.input_spec = [InputSpec(ndim=4)]
        self.output_dim = timestep_out
        self.return_sequences = return_sequences
        self.initial_state = initial_state

        self.supports_masking = False
        self.state_spec = None
        self._states = None
        self.constants_spec = None
        self._num_constants = None

    def compute_output_shape(self, input_shape):

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        cell = self.cell
        if cell.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif cell.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
        rows = conv_utils.conv_output_length(rows,
                                             cell.kernel_size[0],
                                             padding=cell.padding,
                                             stride=cell.strides[0],
                                             dilation=cell.dilation_rate[0])
        cols = conv_utils.conv_output_length(cols,
                                             cell.kernel_size[1],
                                             padding=cell.padding,
                                             stride=cell.strides[1],
                                             dilation=cell.dilation_rate[1])

        output_shape = input_shape[:1] + (self.output_dim, rows, cols, cell.filters)
        output_shape = transpose_shape(output_shape, cell.data_format, spatial_axes=(2, 3))

        if not self.return_sequences:
            output_shape = output_shape[:1] + output_shape[2:]

        return output_shape

    def build(self, input_shape):

        channels = input_shape[-1]
        if self.cell.filters != channels:
            self.cell.filters = channels
        if self.cell.padding != 'same':
            self.cell.padding = 'same'

        batch_size = None
        self.input_spec[0] = InputSpec(shape=(batch_size,) + input_shape[1:4])

        # allow cell (if layer) to build before we set or validate state_spec
        if isinstance(self.cell, Layer):
            step_input_shape = input_shape
            self.cell.build(step_input_shape)

        # set or validate state_spec
        if hasattr(self.cell.state_size, '__len__'):
            state_size = list(self.cell.state_size)
        else:
            state_size = [self.cell.state_size]

        if self.state_spec is not None:
            # initial_state was passed in call, check compatibility
            if self.cell.data_format == 'channels_first':
                ch_dim = 1
            elif self.cell.data_format == 'channels_last':
                ch_dim = 3
            if not [spec.shape[ch_dim] for spec in self.state_spec] == state_size:
                raise ValueError(
                    'An initial_state was passed that is not compatible with '
                    '`cell.state_size`. Received `state_spec`={}; '
                    'However `cell.state_size` is '
                    '{}'.format([spec.shape for spec in self.state_spec],
                                self.cell.state_size))
        else:
            if self.cell.data_format == 'channels_first':
                self.state_spec = [InputSpec(shape=(None, dim, None, None))
                                   for dim in state_size]
            elif self.cell.data_format == 'channels_last':
                self.state_spec = [InputSpec(shape=(None, None, None, dim))
                                   for dim in state_size]
        self.built = True

    def get_initial_state(self, inputs):
        # (samples, timesteps, rows, cols, filters)
        initial_state = K.zeros_like(inputs)
        shape = list(self.cell.kernel_shape)
        shape[-1] = self.cell.filters

        if K.backend() == 'tensorflow':
            # We need to force this to be a tensor
            # and not a variable, to avoid variable initialization
            # issues.
            import tensorflow as tf
            kernel = tf.zeros(tuple(shape))
        else:
            kernel = K.zeros(tuple(shape))
        initial_state = self.cell.input_conv(initial_state,
                                             kernel,
                                             padding=self.cell.padding)
        keras_shape = list(K.int_shape(inputs))
        if K.image_data_format() == 'channels_first':
            indices = 2, 3
        else:
            indices = 1, 2
        for i, j in enumerate(indices):
            keras_shape[j] = conv_utils.conv_output_length(
                keras_shape[j],
                shape[i],
                padding=self.cell.padding,
                stride=self.cell.strides[i],
                dilation=self.cell.dilation_rate[i])
        initial_state._keras_shape = keras_shape

        if hasattr(self.cell.state_size, '__len__'):
            return [initial_state for _ in self.cell.state_size]
        else:
            return [initial_state]

    def call(self, inputs, training=None, initial_state=None, **kwargs):
        initial_state = self.initial_state
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            inputs = inputs[0]
        if initial_state is not None:
            pass
        else:
            initial_state = self.get_initial_state(inputs)

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')

        kwargs = {}
        if has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        def step(inputs, states):
            return self.cell.call(inputs, states)

        uses_learning_phase = False

        shape = K.int_shape(inputs)
        states = tuple(initial_state)
        outputs = []
        current = inputs
        i = 0

        while i < self.output_dim:

            output, new_states = step(current, tuple(states))

            if getattr(output, '_uses_learning_phase', False):
                uses_learning_phase = True

            outputs.append(output)
            states = new_states[:len(states)]
            current = output
            i += 1

        outputs = K.stack(outputs, 1)

        if not self.return_sequences:
            last_output = outputs[-1]
            # Properly set learning phase
            if getattr(last_output, '_uses_learning_phase', False):
                last_output._uses_learning_phase = True
            return last_output

        return outputs

    @property
    def filters(self):
        return self.cell.filters

    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint':
                      constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvRNN2D_readout, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))
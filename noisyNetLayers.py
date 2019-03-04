from keras import backend as K
from keras.layers import Dense, conv_utils, activations, regularizers, constraints
from keras.layers.convolutional import _Conv, Conv2D
from keras.engine.base_layer import InputSpec
from keras import initializers


class NoisyDense(Dense):
    def __init__(self, units, **kwargs):
        self.output_dim = units
        super(NoisyDense, self).__init__(units, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=None,
                                      constraint=None)

        self.kernel_sigma = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=initializers.Constant(0.017),
                                      name='sigma_kernel',
                                      regularizer=None,
                                      constraint=None)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=None,
                                        constraint=None)

            self.bias_sigma = self.add_weight(shape=(self.units,),
                                        initializer=initializers.Constant(0.017),
                                        name='bias_sigma',
                                        regularizer=None,
                                        constraint=None)
        else:
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})
        self.built = True

    def call(self, inputs):
        print(inputs.get_shape().as_list() + [self.units])
        self.kernel_epsilon = K.random_normal(shape=(self.input_dim, self.units))

        w = self.kernel + K.tf.multiply(self.kernel_sigma, self.kernel_epsilon)
        output = K.dot(inputs, w)

        if self.use_bias:
            self.bias_epsilon = K.random_normal(shape=(self.units,))

            b = self.bias + K.tf.multiply(self.bias_sigma, self.bias_epsilon)
            output = output + b
        if self.activation is not None:
            output = self.activation(output)
        return output


class NoisyConv2D(Conv2D):

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        self.input_dim = input_shape[channel_axis]
        self.kernel_shape = self.kernel_size + (self.input_dim, self.filters)

        self.kernel = self.add_weight(shape=self.kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.kernel_sigma = self.add_weight(shape=self.kernel_shape,
                                      initializer=initializers.Constant(0.017),
                                      name='kernel_sigma',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

            self.bias_sigma = self.add_weight(shape=(self.filters,),
                                        initializer=initializers.Constant(0.017),
                                        name='bias_sigma',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: self.input_dim})
        self.built = True

    def call(self, inputs):
        # add noise to kernel

        self.kernel_epsilon = K.random_normal(shape=self.kernel_shape)

        w = self.kernel + K.tf.multiply(self.kernel_sigma, self.kernel_epsilon)

        # do conv op
        outputs = K.conv2d(
            inputs,
            w,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            self.bias_epsilon = K.random_normal(shape=(self.filters,))

            b = self.bias + K.tf.multiply(self.bias_sigma, self.bias_epsilon)
            outputs = K.bias_add(
                outputs,
                b,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

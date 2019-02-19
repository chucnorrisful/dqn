from keras import backend as K
from keras.layers import Dense, conv_utils, activations, regularizers, constraints
from keras.layers.convolutional import _Conv
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
                                      initializer=initializers.get('lecun_uniform'),
                                      name='kernel',
                                      regularizer=None,
                                      constraint=None)

        self.kernel_sigma = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=initializers.Constant(0.017),
                                      name='sigma_kernel',
                                      regularizer=None,
                                      constraint=None)

        # self.kernel_epsilon = self.add_weight(shape=(input_dim, self.units),
        #                               initializer=initializers.get('zeros'),
        #                               name='kernel_epsilon',
        #                               regularizer=None,
        #                               constraint=None)

        self.bias = self.add_weight(shape=(self.units,),
                                    initializer=initializers.get('lecun_uniform'),
                                    name='bias',
                                    regularizer=None,
                                    constraint=None)

        self.bias_sigma = self.add_weight(shape=(self.units,),
                                    initializer=initializers.Constant(0.017),
                                    name='bias_sigma',
                                    regularizer=None,
                                    constraint=None)

        # self.bias_epsilon = self.add_weight(shape=(self.units,),
        #                             initializer=initializers.get('zeros'),
        #                             name='bias_epsilon',
        #                             regularizer=None,
        #                             constraint=None)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})
        self.built = True

    def call(self, inputs):
        print(inputs.get_shape().as_list() + [self.units])
        self.kernel_epsilon = K.random_normal(shape=(self.input_dim, self.units))

        self.bias_epsilon = K.random_normal(shape=(self.units,))

        w = self.kernel + K.tf.multiply(self.kernel_sigma, self.kernel_epsilon)
        output = K.dot(inputs, w)

        b = self.bias + K.tf.multiply(self.bias_sigma, self.bias_epsilon)
        output = output + b
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


class NoisyConv2D(_Conv):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(_Conv, self).__init__(**kwargs)
        self.rank = 2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank,
                                                      'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, self.rank,
                                                        'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

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
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: self.input_dim})
        self.built = True

    def call(self, inputs):

        outputs = K.conv2d(
            inputs,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import InputSpec
class PConv3D(Conv3D):
    def __init__(self, *args, n_channels=3, mono=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [InputSpec(ndim=5), InputSpec(ndim=5)]

    def build(self, input_shape):

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1


        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')

        self.input_dim = input_shape[0][channel_axis]

        # Seismic data kernel
        kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='seismic_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # Mask kernel
        self.kernel_mask = K.ones(self.kernel_size + (self.input_dim, self.filters))

        # Calculate padding size to achieve zero-padding
        self.pconv_padding = (
            (int((self.kernel_size[0] - 1) / 2), int((self.kernel_size[0] - 1) / 2)),
            (int((self.kernel_size[0] - 1) / 2), int((self.kernel_size[0] - 1) / 2)),
            (int((self.kernel_size[0] - 1) / 2), int((self.kernel_size[0] - 1) / 2)),
        )


        # Window size - used for normalization
        self.window_size = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):

        # Both seismic data and mask must be supplied
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception(
                'PartialConvolution3D must be called on a list of two tensors [seismic, mask]. Instead got: ' + str(inputs))

        # Padding done explicitly so that padding becomes part of the masked partial convolution
        seismic = K.spatial_3d_padding(inputs[0], self.pconv_padding, self.data_format)
        masks = K.spatial_3d_padding(inputs[1], self.pconv_padding, self.data_format)

        # Apply convolutions to mask
        mask_output = K.conv3d(
            masks, self.kernel_mask,
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Apply convolutions to seismic data
        seis_output = K.conv3d(
            (seismic * masks), self.kernel,
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        mask_ratio = self.window_size / (mask_output + 1e-8)

        # Clip output to be between 0 and 1
        mask_output = K.clip(mask_output, 0, 1)

        mask_ratio = mask_ratio * mask_output
        seis_output = seis_output * mask_ratio

        if self.use_bias:
            seis_output = K.bias_add(
                seis_output,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            seis_output = self.activation(seis_output)

        return [seis_output, mask_output]



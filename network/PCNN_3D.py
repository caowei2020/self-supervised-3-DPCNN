import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from network.pconv_layer_3d import PConv3D
import math

def get_weight(shape, gain=np.sqrt(2)):
    fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)
    w = tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))
    return w

def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[-1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    return x + tf.reshape(b, [1, 1, 1, 1,-1])

def Pconv3d_bias(x, fmaps, kernel, mask_in=None):
    assert kernel >= 1 and kernel % 2 == 1
    conv, mask = PConv3D(fmaps, kernel, strides=1, padding='valid',
                         data_format='channels_last')([x, mask_in])
    return conv, mask

def Conv3d_bias(x, fmaps, kernel, gain=np.sqrt(2)):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([1, kernel, kernel, x.shape[-1].value, fmaps], gain=gain)
    w = tf.cast(w, x.dtype)
    x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1],[0,0]], "SYMMETRIC")
    return apply_bias(tf.nn.conv3d(x, w, strides=[1, 1, 1, 1,1], padding='VALID', data_format='NDHWC'))


def Pmaxpool3d(x, k=2, mask_in=None):
    ksize = [1, k, k, k, 1]
    x = tf.nn.max_pool3d(x, ksize=ksize, strides=ksize, padding='SAME', data_format='NDHWC')
    mask_out = tf.nn.max_pool3d(mask_in, ksize=ksize, strides=ksize, padding='SAME', data_format='NDHWC')
    return x, mask_out

def Upscale3d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale3D'):
        return tf.keras.layers.UpSampling3D(size=(factor, factor, factor), data_format='channels_last')(x)

def Conv_3d_lr(name, x, fmaps):
    with tf.variable_scope(name):
        return tf.nn.leaky_relu(Conv3d_bias(x, fmaps, 3), alpha=0.1)

def Pconv_3d_lr_dp(name, x, fmaps, mask_in, dp):
    with tf.variable_scope(name):
        x = tf.nn.dropout(x, rate=dp)           # "rate" is dropout rate, rate=1-keep_prob
        x_out, mask_out = Pconv3d_bias(x, fmaps, 3, mask_in=mask_in)
        return tf.nn.leaky_relu(x_out, alpha=0.1), mask_out

def Pconv_3d_lr(name, x, fmaps, mask_in):
    with tf.variable_scope(name):
        x_out, mask_out = Pconv3d_bias(x, fmaps, 3, mask_in=mask_in)
        return tf.nn.leaky_relu(x_out, alpha=0.1), mask_out

def partial_conv_net_3d(x, mask, depth, height, width, channel, dp, **_kwargs):
    x.set_shape([None, depth, height, width, channel])
    mask.set_shape([None, depth, height, width, channel])
    skips = [x]
    skips_mask = [mask]
    # encoder
    n = x
    n, mask = Pconv_3d_lr('e_conv1a', n, 32, mask_in=mask)
    n, mask = Pconv_3d_lr('e_conv1b', n, 32, mask_in=mask)
    n, mask = Pmaxpool3d(n, mask_in=mask)
    skips.append(n)
    skips_mask.append(mask)

    n, mask = Pconv_3d_lr('e_conv2a', n, 32, mask_in=mask)
    n, mask = Pconv_3d_lr('e_conv2b', n, 32, mask_in=mask)
    n, mask = Pmaxpool3d(n, mask_in=mask)
    skips.append(n)
    skips_mask.append(mask)

    n, mask = Pconv_3d_lr('e_conv3a', n, 32, mask_in=mask)
    n, mask = Pconv_3d_lr('e_conv3b', n, 32, mask_in=mask)
    n, mask = Pmaxpool3d(n, mask_in=mask)
    skips.append(n)
    skips_mask.append(mask)

    n, mask = Pconv_3d_lr('e_conv4a', n, 32, mask_in=mask)
    n, mask = Pconv_3d_lr('e_conv4b', n, 32, mask_in=mask)
    n, mask = Pmaxpool3d(n, mask_in=mask)

    n, mask = Pconv_3d_lr('e_conv5a', n, 32, mask_in=mask)
    n, mask = Pconv_3d_lr('e_conv5b', n, 32, mask_in=mask)

    # decoder
    n = Upscale3d(n)
    mask = Upscale3d(mask)
    n = Concat_3d(n, skips.pop())
    mask = Concat_3d(mask, skips_mask.pop())

    n, mask = Pconv_3d_lr_dp('d_conv1a', n, 32, mask_in=mask,dp=dp)
    n, mask = Pconv_3d_lr_dp('d_conv1b', n, 32, mask_in=mask,dp=dp)
    n, mask = Pconv_3d_lr_dp('d_conv1c', n, 32, mask_in=mask,dp=dp)

    n = Upscale3d(n)
    mask = Upscale3d(mask)
    n = Concat_3d(n, skips.pop())
    mask = Concat_3d(mask, skips_mask.pop())

    n, mask = Pconv_3d_lr_dp('d_conv2a', n, 32, mask_in=mask,dp=dp)
    n, mask = Pconv_3d_lr_dp('d_conv2b', n, 32, mask_in=mask,dp=dp)
    n, mask = Pconv_3d_lr_dp('d_conv2c', n, 32, mask_in=mask,dp=dp)

    n = Upscale3d(n)
    mask = Upscale3d(mask)
    n = Concat_3d(n, skips.pop())
    mask = Concat_3d(mask, skips_mask.pop())
    n, mask = Pconv_3d_lr_dp('d_conv3a', n, 32, mask_in=mask,dp=dp)
    n, mask = Pconv_3d_lr_dp('d_conv3b', n, 32, mask_in=mask,dp=dp)
    n, mask = Pconv_3d_lr_dp('d_conv3c', n, 32, mask_in=mask, dp=dp)

    n = Upscale3d(n)
    mask = Upscale3d(mask)
    n = Concat_3d(n, skips.pop())
    mask = Concat_3d(mask, skips_mask.pop())
    n, mask = Pconv_3d_lr_dp('d_conv4a', n, 32, mask_in=mask,dp=dp)
    n, mask = Pconv_3d_lr_dp('d_conv4b', n, 32, mask_in=mask,dp=dp)
    n, mask = Pconv_3d_lr_dp('d_conv4c', n, 32, mask_in=mask, dp=dp)

    n = Conv_3d_lr('d_conv5a', n, 32)
    n = Conv_3d_lr('d_conv5b', n, 32)
    n = Conv_3d_lr('d_conv5c', n, channel)

    return n


def Concat_3d(x, y):
    bs1, d1, h1, w1, c1 = x.shape.as_list()
    bs2, d2, h2, w2, c2 = y.shape.as_list()
    x = tf.keras.layers.Cropping3D(cropping=((0,0) if d1 <d2 else (math.floor((d1-d2)/2),math.ceil((d1-d2)/2)) if (d1-d2) % 2 == 1 else (int((d1-d2)/2),int((d1-d2)/2)), \
                                             (0,0) if h1 <h2 else (math.floor((h1-h2)/2),math.ceil((h1-h2)/2)) if (h1-h2) % 2 == 1 else (int((h1-h2)/2),int((h1-h2)/2)), \
                                             (0,0) if w1 <w2 else (math.floor((w1-w2)/2),math.ceil((w1-w2)/2)) if (w1-w2) % 2 == 1 else (int((w1-w2)/2),int((w1-w2)/2))), \
                                             data_format='channels_last')(x)


    y = tf.keras.layers.Cropping3D(cropping=((0,0) if d1 >d2 else (math.floor((d2-d1)/2),math.ceil((d2-d1)/2)) if (d2-d1) % 2 == 1 else (int((d2-d1)/2),int((d2-d1)/2)),\
                                             (0,0) if h1 >h2 else (math.floor((h2-h1)/2),math.ceil((h2-h1)/2)) if (h2-h1) % 2 == 1 else (int((h2-h1)/2),int((h2-h1)/2)),\
                                             (0,0) if w1 >w2 else (math.floor((w2-w1)/2),math.ceil((w2-w1)/2)) if (w2-w1) % 2 == 1 else (int((w2-w1)/2),int((w2-w1)/2))),\
                                             data_format='channels_last')(y)
    return tf.concat([x, y], axis=4)


def build_denoising_net(noisy, nn_dp, data_mp):
    _, d, h, w, c = np.shape(noisy)
    noisy_tensor = tf.identity(noisy)
    is_flip_lr = tf.placeholder(tf.int16)
    is_flip_ud = tf.placeholder(tf.int16)
    is_reverse_polarity=tf.placeholder(tf.int16)
    noisy_tensor = data_arg_3d_8(noisy_tensor, is_flip_lr, is_flip_ud, is_reverse_polarity)
    response = noisy_tensor
    mask_tensor = tf.ones_like(response)
    mask_tensor = tf.nn.dropout(mask_tensor, rate=data_mp) * (1-data_mp)
    response = tf.multiply(mask_tensor, response)
    response = partial_conv_net_3d(response, mask_tensor, depth=d, height=h,width=w,channel=c, dp=nn_dp)
    data_loss = mask_loss(response, noisy_tensor, 1. - mask_tensor)
    response = data_arg_3d_8(response, is_flip_lr, is_flip_ud, is_reverse_polarity)
    our_volume = response

    saver = tf.train.Saver(max_to_keep=1)
    model = {
        'training_error': data_loss,
        'saver': saver,
        'our_volume': our_volume,
        'is_flip_lr': is_flip_lr,
        'is_flip_ud': is_flip_ud,
        'is_reverse_polarity': is_reverse_polarity,
    }
    return model

def build_reconstruction_net(volume, missed, task_name, nn_dp=0.35, data_mp=0.35):
    _, d, h, w, c = np.shape(volume)
    volume_tensor = tf.identity(volume)
    missed_tensor = tf.identity(missed)
    is_reverse_polarity = tf.placeholder(tf.int16)
    volume_tensor = data_arg_3d_2(volume_tensor, is_reverse_polarity)
    response = volume_tensor
    mask_tensor = missed_tensor
    mask_tensor = tf.nn.dropout(mask_tensor, rate=data_mp) * (1-data_mp)
    response = tf.multiply(mask_tensor, response)
    response = partial_conv_net_3d(response, mask_tensor, depth=d, height=h, width=w, channel=c, dp=nn_dp)
    data_loss = mask_loss(response, volume_tensor, missed_tensor - mask_tensor)
    if task_name == 'R':
        our_volume = volume_tensor + tf.multiply(response, 1 - missed_tensor)
    else:
        our_volume = response

    our_volume = data_arg_3d_2(our_volume, is_reverse_polarity)

    saver = tf.train.Saver(max_to_keep=1)
    model = {
        'training_error': data_loss,
        'saver': saver,
        'our_volume': our_volume,
        'is_reverse_polarity': is_reverse_polarity,
    }
    return model

def mask_loss(x, labels, masks):
    cnt_nonzero = tf.cast(tf.count_nonzero(masks), x.dtype)
    loss = tf.reduce_sum(tf.multiply(tf.math.pow(x - labels, 2), masks)) / cnt_nonzero
    return loss

def data_arg_3d_2(x,is_reverse_polarity):
    x = tf.squeeze(x,axis=0)
    x = tf.cond(is_reverse_polarity > 0, lambda: x *(-1), lambda: x)
    x = tf.expand_dims(x,axis=0)
    return x

def data_arg_3d_4(x, is_flip_lr, is_flip_ud):
    x = tf.squeeze(x, axis=0)
    x = tf.cond(is_flip_lr > 0, lambda: tf.image.flip_left_right(x), lambda: x)
    x = tf.cond(is_flip_ud > 0, lambda: tf.image.flip_up_down(x), lambda: x)
    x = tf.expand_dims(x, axis=0)
    return x

def data_arg_3d_8(x, is_flip_lr, is_flip_ud,is_reverse_polarity):
    x = tf.squeeze(x,axis=0)
    x = tf.cond(is_flip_lr > 0, lambda: tf.image.flip_left_right(x), lambda: x)
    x = tf.cond(is_flip_ud > 0, lambda: tf.image.flip_up_down(x), lambda: x)
    x = tf.cond(is_reverse_polarity > 0, lambda: x *(-1), lambda: x)
    x = tf.expand_dims(x,axis=0)
    return x
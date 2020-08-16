import tensorflow as tf

class SpectralNormalization(tf.keras.layers.Layer):
  def __init__(self, layer,
               kernel_name = 'kernel',
               input_dims = [0,1,2],
               output_dims = [3],
               iteration = 1):
    super().__init__()
    self.layer = layer
    self.kernel_name = kernel_name
    self.input_dims = input_dims
    self.output_dims = output_dims
    self.iteration = iteration

  def build(self, input_shape):
    self.layer.build(input_shape)
    self.kernel = getattr(self.layer, self.kernel_name)
    shape = self.kernel.shape.as_list()
    u_dim = 1
    for i in self.output_dims:
      u_dim *= shape[i]
    self.u = self.add_weight("u", [1, u_dim],
                             initializer = tf.initializers.TruncatedNormal(),
                             trainable = False)
    self.u_dim = u_dim

  def call(self, x, training=None):
    if training:
      self.normalize_kernel()
    return self.layer.call(x)

  def normalize_kernel(self):
    w = tf.transpose(self.kernel, self.input_dims+self.output_dims)
    w = tf.reshape(w, [-1, self.u_dim])

    u_hat = self.u
    v_hat = None
    for i in range(self.iteration): # power iteration
      v_ = tf.matmul(u_hat, w, transpose_b=True)
      v_hat = tf.nn.l2_normalize(v_)
      u_ = tf.matmul(v_hat, w)
      u_hat = tf.nn.l2_normalize(u_)

    #u_hat = tf.stop_gradient(u_hat)
    #v_hat = tf.stop_gradient(v_hat)
    self.u.assign(u_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), u_hat, transpose_b=True)
    new_kernel = tf.reshape(w / sigma, self.kernel.shape)
    self.kernel.assign(new_kernel)

def create_network(
    n = 5,
    channels = 32,
    output_channels = 3,
    conv_fn = tf.keras.layers.Conv2D,
    depthwise_conv_fn = tf.keras.layers.DepthwiseConv2D,
    bn_scale = True):
  x = tf.keras.Input([None,None,3])
  
  y = x
  y = conv_fn(channels, 3, padding='SAME', use_bias=False)(y)
  y = tf.keras.layers.BatchNormalization(scale=bn_scale)(y)
  y = tf.keras.layers.LeakyReLU()(y)

  for i in range(n-2):
    _y = y
    y = conv_fn(channels*6, 1, padding='SAME', use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization(scale=bn_scale)(y)
    y = tf.keras.layers.LeakyReLU()(y)
    y = depthwise_conv_fn(3, padding='SAME', use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization(scale=bn_scale)(y)
    y = tf.keras.layers.LeakyReLU()(y)
    y = conv_fn(channels, 1, padding='SAME', use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization(scale=bn_scale)(y)
    #y = tf.keras.layers.LeakyReLU()(y)
    y = tf.keras.layers.Dropout(0.2, noise_shape=[None,1,1,1])(y)
    y = y + _y

  y = conv_fn(output_channels, 3, padding='SAME', use_bias=True)(y)

  return tf.keras.Model(x, y)

def create_generator(stage):
  x = tf.keras.Input([None,None,3])
  z = tf.keras.Input([None,None,3])
  y = create_network(
      n = 5,
      channels = 32)(x+z)
  y = tf.math.tanh(y)
  if stage != 0:
    y += x
  return tf.keras.Model([x,z], y)

def create_discriminator(stage):
  def conv_fn(*args, **kwargs):
    conv = tf.keras.layers.Conv2D(*args, **kwargs)
    conv = SpectralNormalization(conv)
    return conv

  def depthwise_conv_fn(*args, **kwargs):
    conv = tf.keras.layers.DepthwiseConv2D(*args, **kwargs)
    conv = SpectralNormalization(conv,
                                 kernel_name = 'depthwise_kernel',
                                 input_dims = [0,1],
                                 output_dims = [2,3])
    return conv

  return create_network(
      n = 5,
      channels = 32,
      output_channels = 1,
      conv_fn = conv_fn,
      depthwise_conv_fn = depthwise_conv_fn,
      bn_scale = False)


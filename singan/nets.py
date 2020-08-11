import tensorflow as tf

class Conv2DSpectralNormalization(tf.keras.layers.Conv2D):
  def __init__(self, *args, **kwargs):
    kwargs['kernel_initializer'] = tf.keras.initializers.TruncatedNormal()
    super().__init__(*args, **kwargs)

  def build(self, input_shape):
    super().build(input_shape)
    shape = self.kernel.shape.as_list()
    self.u = self.add_weight("u", [1, self.filters], initializer=tf.initializers.TruncatedNormal(), trainable=False)
  
  def call(self, x, training=None):
    if training:
      self.kernel.assign(self.spectral_norm())
    return super().call(x)

  def spectral_norm(self, iteration=1):
    w = tf.reshape(self.kernel, [-1, self.filters])

    u_hat = self.u
    v_hat = None
    for i in range(iteration): # power iteration
      v_ = tf.matmul(u_hat, w, transpose_b=True)
      v_hat = tf.nn.l2_normalize(v_)
      u_ = tf.matmul(v_hat, w)
      u_hat = tf.nn.l2_normalize(u_)

    #u_hat = tf.stop_gradient(u_hat)
    #v_hat = tf.stop_gradient(v_hat)
    self.u.assign(u_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), u_hat, transpose_b=True)
    return tf.reshape(w / sigma, self.kernel.shape)

def create_network(
    n = 5,
    channels = 32,
    output_channels = 3,
    conv_fn = tf.keras.layers.Conv2D):
  x = tf.keras.Input([None,None,3])
  
  y = x
  for i in range(n-1):
    y = conv_fn(channels, 3, padding='SAME', use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.LeakyReLU()(y)
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
  return create_network(
      n = 5,
      channels = 32,
      output_channels = 1,
      conv_fn = Conv2DSpectralNormalization)

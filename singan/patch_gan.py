import tensorflow as tf

class PatchGAN(tf.Module):
  def __init__(self, generator, discriminator):
    self.generator = generator
    self.discriminator = discriminator
    self.g_opt = tf.keras.optimizers.Adam(0.0005, beta_1=0.5)
    self.d_opt = tf.keras.optimizers.Adam(0.0005, beta_1=0.5)

  def compute_discriminator_loss(self, input_image, _, noise_image, real_image):
    d_losses = []

    g_inputs = [input_image, noise_image]
    fake_image = self.generator(g_inputs, training=True)

    dis_real = self.discriminator(real_image, training=True)
    dis_fake = self.discriminator(fake_image, training=True)

    # adversarial loss
    adv_loss = tf.reduce_mean(dis_fake - dis_real)
    d_losses += [adv_loss]

    # gradient penalty
    gp_image = real_image
    #alpha = tf.random.uniform([tf.shape(real_image)[0],1,1,1])
    #gp_image = alpha * real_image + (1.-alpha) * fake_image
    with tf.GradientTape() as tape:
      tape.watch(gp_image)
      dis_gp = self.discriminator(gp_image, training=True)
    dis_grads = tape.gradient(dis_gp, gp_image)
    d_losses += [0.1 * tf.reduce_mean(tf.square(dis_grads))]
    #d_losses += [0.1 * tf.reduce_mean(tf.square(tf.math.sqrt(tf.reduce_sum(tf.square(dis_grads),axis=-1))-1))]

    return tf.add_n(self.discriminator.losses + d_losses)

  def compute_generator_loss(self, input_image, rec_image, noise_image, real_image):
    g_losses = []

    g_inputs = [input_image, noise_image]
    fake_image = self.generator(g_inputs, training=True)
    dis_fake = self.discriminator(fake_image, training=True)

    # adversarial loss
    adv_loss = tf.reduce_mean(dis_fake)
    g_losses += [-adv_loss]

    # reconstruction loss
    g_inputs = [rec_image, tf.zeros_like(input_image)]
    reconst_image = self.generator(g_inputs, training=True)
    reconst_loss = tf.reduce_mean(tf.square(reconst_image - real_image))
    g_losses += [10. * reconst_loss]

    return tf.add_n(self.generator.losses + g_losses)

  @tf.function
  def train_step(self, input_image, rec_image, real_image):
    std = tf.reduce_mean(tf.square(input_image - real_image), axis=[1,2,3], keepdims=True)
    noise_image = tf.random.normal(tf.shape(input_image)) * std

    for _ in range(3):
      with tf.GradientTape() as d_tape:
        d_loss = self.compute_discriminator_loss(
            input_image, rec_image, noise_image, real_image)
      d_variables = self.discriminator.trainable_variables
      d_grads = d_tape.gradient(d_loss, d_variables)
      self.d_opt.apply_gradients(zip(d_grads, d_variables))

    for _ in range(3):
      with tf.GradientTape() as g_tape:
        g_loss = self.compute_generator_loss(
            input_image, rec_image, noise_image, real_image)
      g_variables = self.generator.trainable_variables
      g_grads = g_tape.gradient(g_loss, g_variables)
      self.g_opt.apply_gradients(zip(g_grads, g_variables))

    return g_loss, d_loss

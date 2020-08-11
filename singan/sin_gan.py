import tensorflow as tf
from .patch_gan import PatchGAN

class SinGAN(tf.Module):
  def __init__(self,
               generators = None,
               discriminators = None,
               scales = None):
    if scales:
      self.scales = list(scales)
    else:
      self.scales = [(3/4)**(8-i) for i in range(9)]
    if generators:
      self.generators = list(generators)
    else:
      from .nets import create_generator
      self.generators = [create_generator(i) for i,_ in enumerate(self.scales)]
    if discriminators:
      self.discriminators = list(discriminators)
    else:
      from .nets import create_discriminator
      self.discriminators = [create_discriminator(i) for i,_ in enumerate(self.scales)]

    shape = tf.concat([[1],self.default_output_size[0],[3]], axis=0)
    self.seed_image = tf.random.normal(shape)

    self.gans = [PatchGAN(g, d) for g, d in zip(self.generators, self.discriminators)]

  def _scale_size(self, size, scale):
    size = tf.cast(size, tf.float32)
    size = tf.math.round(size*scale)
    size = tf.cast(size, tf.int32)
    return size

  @property
  def default_output_size(self):
    sizes = [self._scale_size([256,256], s) for s in self.scales]
    return sizes
  
  @tf.function
  def generate(self,
               input_stage = 0,
               output_stage = None,
               seed_image = None,
               real_image = None,
               noise_stds = None,
               batch_size = None,
               output_size = None,
               training=None):
    if output_stage is None:
      output_stage = len(self.generators) - 1
    if input_stage < 0 or input_stage >= len(self.generators):
      raise ValueError(f'input_stage must be in range [0, {len(self.generators)}]')
    if output_stage < 0 or output_stage >= len(self.generators):
      raise ValueError(f'output_stage must be in range[0, {len(self.generators)}]')

    if output_size is not None:
      output_size = tf.convert_to_tensor(output_size)
      if output_size.shape.rank == 0:
        output_size = tf.stack([output_size, output_size])
    else:
      output_size = self.default_output_size[output_stage]

    if noise_stds is not None:
      if isinstance(noise_stds, (list, tuple)):
        if len(noise_stds) == output_stage-input_stage+1:
          noise_stds = [tf.reshape(std, [-1,1,1,1]) for std in noise_stds]
        else:
          raise ValueError('len(noise_stds) must be the same as the number of stages to be computed.')
      else:
        noise_stds = [tf.reshape(noise_stds, [-1,1,1,1]) for _ in range(input_stage, output_stage+1)]
    else:
      noise_stds = [None] * (output_stage-input_stage+1)

    if batch_size is None:
      if real_image is not None:
        batch_size = tf.shape(real_image)[0]
      else:
        raise ValueError('batch_size must be specified when real_image is not.')
      
    generators = self.generators[input_stage:output_stage+1]
    discriminators = self.discriminators[input_stage:output_stage+1]
    scales = self.scales[input_stage:output_stage+1]
    scales = [s/scales[-1] for s in scales]
    sizes = [self._scale_size(output_size, s) for s in scales]
    shapes = [tf.concat([[batch_size], size, [3]], axis=0) for size in sizes]

    if seed_image is not None:
      image = seed_image
    else:
      image = tf.tile(self.seed_image, [batch_size,1,1,1])

    #print(f'Starting with image of size {image.shape[1:3]}')
    for i, (generator, discriminator, size, shape, std) in enumerate(zip(generators, discriminators, sizes, shapes, noise_stds)):
      image = tf.image.resize(image, size)
      if real_image is not None:
        std = tf.reduce_mean(tf.square(image - tf.image.resize(real_image, size)), axis=[1,2,3], keepdims=True)
      if std is not None:
        noise_image = tf.random.normal(shape) * std
      else:
        noise_image = tf.zeros(shape)
      #print(f'Generating image of size {size}' + (f' with noise of std {std}' if std is not None else ''))
      image = generator([image, noise_image], training=training)
    return image

  def train_stage(self, stage, real_image, steps=2000):
    #generator = self.generators[stage]
    #discriminator = self.discriminators[stage]
    #gan = PatchGAN(generator, discriminator)
    gan = self.gans[stage]
    batch_size = tf.shape(real_image)[0]

    orig_size = tf.shape(real_image)[1:3]
    size = self._scale_size(orig_size,self.scales[stage])

    if stage == 0:
      seed_image = tf.tile(self.seed_image, [batch_size,1,1,1])
      rec_image = seed_image
    else:
      prev_size = self._scale_size(orig_size,self.scales[stage-1])
      rec_image = self.generate(
          input_stage = 0,
          output_stage = stage-1,
          batch_size = batch_size,
          output_size = prev_size,
          training = False
      )
      rec_image = tf.image.resize(rec_image, size)
    real_image = tf.image.resize(real_image, size)

    for i in range(steps):
      if stage != 0:
        seed_image = self.generate(
            input_stage = 0,
            output_stage = stage-1,
            real_image = real_image,
            output_size = prev_size,
            training = False
        )
        seed_image = tf.image.resize(seed_image, size)
      g_loss, d_loss = gan.train_step(seed_image, rec_image, real_image)
      print(f'\r[{i+1}/{steps}] generator_loss: {g_loss}, discriminator_loss: {d_loss}', end='')
    print()

  def _create_ckpt(self):
    save_dict = {}
    for i, gan in enumerate(self.gans):
      save_dict[f'generator_{i}'] = gan.generator
      save_dict[f'generator_opt_{i}'] = gan.g_opt
      save_dict[f'discriminator_{i}'] = gan.discriminator
      save_dict[f'discriminator_opt_{i}'] = gan.d_opt
    return tf.train.Checkpoint(**save_dict)

  def save(self, save_dir):
    import os
    ckpt = self._create_ckpt()
    ckpt.save(os.path.join(save_dir, 'singan'))

  def restore(self, save_dir):
    ckpt = self._create_ckpt()
    ckpt.restore(tf.train.latest_checkpoint(save_dir))

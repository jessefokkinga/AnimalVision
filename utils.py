import tensorflow as tf

def load_and_prep_image(filename, img_shape=224, rescale=False):

  img = tf.io.decode_image(filename, channels=3) 
  img = tf.image.resize(img, [img_shape, img_shape])
  if rescale:
      return img/255.
  else:
      return img

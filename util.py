import tensorflow as tf
import keras

def make_prediction(image, model, class_names):
    image = load_and_prep_image(image)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    model = keras.models.load_model('models\\' + model + '.h5')
    preds = model.predict(image)
    pred_class = class_names[tf.argmax(preds[0])]
    pred_conf = int(tf.reduce_max(preds[0])*100)
    return image, pred_class, pred_conf

def load_and_prep_image(filename, img_shape=224, rescale=False):

  img = tf.io.decode_image(filename, channels=3) 
  img = tf.image.resize(img, [img_shape, img_shape])
  if rescale:
      return img/255.
  else:
      return img
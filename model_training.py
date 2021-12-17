import zipfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

def unzip_data(filename):

    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall()
    zip_ref.close()
    
def create_data_loaders(train_dir, image_size=(224, 224)):

    train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                  label_mode="categorical",
                                                                  image_size=image_size)

    return train_data

def create_model(input_shape, base_model, num_classes):    

    data_augmentation = keras.Sequential([
      preprocessing.RandomFlip("horizontal"),
      preprocessing.RandomRotation(0.2),
      preprocessing.RandomZoom(0.2),
      preprocessing.RandomHeight(0.2),
      preprocessing.RandomWidth(0.2),
    ], name ="data_augmentation")
        
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_layer")

    x = data_augmentation(inputs)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output_layer")(x)

    model = keras.Model(inputs, outputs)

    model.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

    return model
    
def load_and_prep_image(filename, img_shape=224, scale=False):

    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [img_shape, img_shape])
    
    if scale:
        return img/255.
    else:
        return img
        
def main(model_path = 'animal_classification_model.h5'):

    INPUT_SHAPE = (224, 224, 3)
    BASE_MODEL = tf.keras.applications.EfficientNetB0(include_top=False)
    
    train_data = create_data_loaders(train_dir="images")
    model = create_model(input_shape=INPUT_SHAPE, base_model=BASE_MODEL, num_classes=len(train_data.class_names))

    model.fit(train_data,
                        epochs=1,
                        steps_per_epoch=len(train_data),
                        validation_data=train_data,
                        validation_steps=int(0.25 * len(train_data)))
                        
    save_model(model, model_path)                    

def save_model(model, path):

    model.save('models/' + path)
    
if __name__ == "__main__":

    main()
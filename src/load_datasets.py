
from config import * 
from models import *

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# input_shape : Optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (299, 299, 3) (with channels_last data format) or (3, 299, 299) (with 
# include_top : Boolean, whether to include the fully-connected layer at the top, as the last layer of the network. Default to True.
# weights     : One of None (random initialization), imagenet (pre-training on ImageNet), or the path to the weights file to be loaded. Default to imagenet.
# pooling     : Optional pooling mode for feature extraction when include_top is False.
#               avg means that global average pooling will be applied to the output of the last convolutional block, and thus the output of the model will be a 2D tensor.
# classes     : optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified. Default to 1000.


def preprocess_image(image_path, img_size):
    """
        Load image from filepath and preprocess using inception_v3.preprocess_input
    """
    
    img = image.load_img(image_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    return(img_array)


def image_process(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return(image, label)


def read_images_dataset(dataset_path, batch_size, img_size):
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path + 'TRAIN', 
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=42)
    
    
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path + 'TEST', 
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=42)
    
    return(train_dataset, validation_dataset)
    


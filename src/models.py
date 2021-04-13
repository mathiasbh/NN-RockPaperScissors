import sys
sys.path.append('..')
from config import *


def create_model(learning_rate, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metric=tf.keras.metrics.SparseCategoricalAccuracy(), img_shape=IMG_SHAPE):
    """
        Sequential model, CNN
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, 11, strides=3, padding='same', activation='relu', input_shape=IMG_SHAPE))
    model.add(tf.keras.layers.AveragePooling2D(2, 3))
    model.add(tf.keras.layers.Conv2D(16, 3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(128, 3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(N_CLASSES, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss=loss, metrics=[metric])
    return(model)


def setup_model(base_model, learning_rate, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metric=tf.keras.metrics.SparseCategoricalAccuracy(), img_shape=IMG_SHAPE):
    base_model.trainable = False
    model = Wrapper(base_model, img_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss=loss, metrics=[metric])
    return(model)



class Wrapper(tf.keras.Model):
    """
        Additional layers to pretrained model
    """
    
    def __init__(self, base_model, img_shape):
        super(Wrapper, self).__init__()
        self.base_model = base_model
        self.average_pooling_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.25)
        self.dense3 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(N_CLASSES, activation='softmax')
        
        
    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.average_pooling_layer(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.output_layer(x)
        return(x)




def load_googlenet(IMG_SHAPE, learning_rate):
    googlenet_base = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    googlenet = setup_model(googlenet_base, learning_rate)
    return(googlenet)
    
    
def load_vggnet(IMG_SHAPE, learning_rate):
    vgg16_base = tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    vgg16 = setup_model(vgg16_base, learning_rate)
    return(vgg16)

def load_resnet(IMG_SHAPE, learning_rate):
    resnet_base = tf.keras.applications.ResNet101V2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    resnet = setup_model(resnet_base, learning_rate)
    return(resnet)



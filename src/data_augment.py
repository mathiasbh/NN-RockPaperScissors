
from config import *
import os 
from tensorflow.keras.preprocessing import image

FOLDER_INPUT = ['PAPER', 'ROCK', 'SCISSOR']
PATH = 'D:/Google drive/Coding/MachineLearning/NN-RockPaperScissors/_DATA/'


# Data augmentation of both data sets
# https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
# https://www.tensorflow.org/hub/tutorials/tf2_image_retraining



datagen_train = image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)


datagen_test = image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)


FOLDER_INPUT = ['PAPER', 'ROCK', 'SCISSOR']

for cat in FOLDER_INPUT:
    path_train = os.path.join(DATASET_PATH,'TRAIN',cat)
    path_test = os.path.join(DATASET_PATH,'TEST',cat)

    ds_train = datagen_train.flow_from_directory(path_train, target_size=(192,192), save_format='jpg', save_to_dir=path_train, save_prefix='aug', class_mode='sparse')
    ds_test = datagen_test.flow_from_directory(path_test, target_size=(192,192), save_format='jpg', save_to_dir=path_test, save_prefix='aug', class_mode='sparse')
    for i in range(4):
        ds_train.next()
        #ds_test.next()





### plotting examples
# import matplotlib.pyplot as plt

# image, label = next(iter(train_ds))
# _ = plt.imshow(image)
# #_ = plt.title(get_label_name(label))


# image = tf.expand_dims(image, 0)

# plt.figure(figsize=(10, 10))
# for i in range(16):
#     augmented_image = data_augmentation(image)
#     ax = plt.subplot(4, 4, i + 1)
#     plt.imshow(augmented_image[0])
#     plt.axis("off")


# plt.show()
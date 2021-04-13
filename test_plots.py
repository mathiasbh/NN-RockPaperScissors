
import sys
sys.path.append('src')
from config import *
from load_datasets import *
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

PATH = 'D:/Google drive/Coding/MachineLearning/NN-RockPaperScissors/_MODELS/saved_model_googlenet_5'

DATASET_NAME = 'rock_paper_scissors'
BATCH_SIZE = 1


# Load dataset
dataset_test_raw, dataset_info = tfds.load(
    name=DATASET_NAME,
    data_dir='tmp',
    with_info=True,
    as_supervised=True,
    split=tfds.Split.TEST,
)

# Function to convert label ID to labels string.
get_label_name = dataset_info.features['label'].int2str
dataset_test = dataset_test_raw.map(image_process)
#dataset_test = dataset_test.take(1)




model = tf.keras.models.load_model(PATH)

# COUNT WRONGLY PREDICTED
# j = 0
# for image, label in dataset_test:
#     image_label = get_label_name(label)
#     model_label = get_label_name(np.argmax(model([image])))
#     if image_label != model_label:
#         j = j + 1


# PLOT WRONGLY PREDICTED IMAGES
plt.figure(figsize=(16, 16))
imgrid = 6 # 6x6 images
i = 0
for image, label in dataset_test:
    image_label = get_label_name(label)
    model_label = get_label_name(np.argmax(model([image])))
    if image_label != model_label:
        ax = plt.subplot(imgrid, imgrid, i + 1)
        _ = plt.imshow(image)
        _ = ax.set_xlabel('Actual/Model: ' + image_label + ' / ' + model_label, fontsize=8)
        _ = ax.axes.xaxis.set_ticks([])
        _ = ax.axes.yaxis.set_ticks([])
        i = i + 1
        if i > imgrid*imgrid - 1:
            break


plt.show()

import sys
sys.path.append('src')

from config import *
from models import load_googlenet, create_model
from load_datasets import *

import tensorflow_datasets as tfds


learning_rate = 0.00005
EXPORT_PATH = '_MODELS/saved_model_googlenet_6'


data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, fill_mode='constant'),
  tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0.2, fill_mode='constant'),
  tf.keras.layers.experimental.preprocessing.RandomZoom(0.25, fill_mode='constant'),
  tf.keras.layers.experimental.preprocessing.RandomContrast(0.4),
])



def prepare(ds, shuffle=False, batch=False, batch_size=BATCH_SIZE, augment=False):
    if shuffle:
        ds = ds.shuffle(1000)
    
    # Batch all datasets
    if batch:
        ds = ds.batch(batch_size)
    
    # Use data augmentation only on the training set
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    
    # Use buffered prefecting on all datasets
    return(ds.prefetch(buffer_size=AUTOTUNE))



# Load model and data // create_model(learning_rate) for self-defined model
model = load_googlenet(IMG_SHAPE, learning_rate)

(dataset_train_raw, dataset_test_raw), dataset_info = tfds.load(
    name=DATASET_NAME,
    data_dir='tmp',
    with_info=True,
    as_supervised=True,
    split=[tfds.Split.TRAIN, tfds.Split.TEST],
)

NUM_TRAIN_EXAMPLES = dataset_info.splits['train'].num_examples
NUM_TEST_EXAMPLES = dataset_info.splits['test'].num_examples
NUM_CLASSES = dataset_info.features['label'].num_classes

INPUT_IMG_SIZE_ORIGINAL = dataset_info.features['image'].shape[0]
INPUT_IMG_SHAPE_ORIGINAL = dataset_info.features['image'].shape

# Function to convert label ID to labels string.
get_label_name = dataset_info.features['label'].int2str

# Preprocess (normalize)
dataset_train = dataset_train_raw.map(image_process)
dataset_test = dataset_test_raw.map(image_process)

# Batch, data augment, and shuffle
dataset_train = prepare(dataset_train, shuffle=True, batch=True, augment=True)
dataset_test = prepare(dataset_test, shuffle=True, batch=True, batch_size=1)



# Create data set from file directory
#dataset_train, dataset_test = read_images_dataset(DATASET_PATH, BATCH_SIZE, img_size=IMG_SIZE)
#dataset_train = prepare(dataset_train, augment=True)
#dataset_test = prepare(dataset_test, shuffle=True)



# Train, test and save model
model.fit(dataset_train, epochs=250)
results = model.evaluate(dataset_test)
print(results)
model.save(EXPORT_PATH)
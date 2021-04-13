
import sys
sys.path.append('src')

import numpy as np
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
    
    
    
DATASET_PATH = 'D:/Google drive/Coding/MachineLearning/NN-RockPaperScissors_data/_DATA/'
#DATASET_PATH = 'D:/Google drive/Coding/MachineLearning/NN-RockPaperScissors_data/_DATA/TRAIN/'
DATASET_NAME = 'rock_paper_scissors'

    
N_CLASSES = 3 # CHANGE HERE, total number of classes
CHANNELS = 3
#IMG_SIZE = 350
IMG_SIZE = 192
BATCH_SIZE = 32
SHUFFLE_SIZE = 1000
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, CHANNELS)
AUTOTUNE = tf.data.experimental.AUTOTUNE

IMAGE_CLASS = {"0": "rock", "1": "paper", "2": "scissors"}

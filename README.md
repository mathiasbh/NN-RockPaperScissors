# Rock Paper Scissors classification with GoogLeNet
Image classifier (3 classes) using GoogLeNet

## Introduction and data
In this project I use the pretrained model GoogLeNet to make a rock-paper-scissors classifier as well as using the setup to play around with my own model setup.
I try two different datasets, the TensorFlow dataset [rock_paper_scissors](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors) and a (smaller) dataset I have constructed myself. Ultimately, the shown results use former as they gave the best results.

I use data augmentation (flip, rotation, translation, zoom, constrast) to increase the dataset size and generalize the model a bit better.

Finally, I use openCV to predict hand motions live from a webcam.


## Results
Using GoogLeNet I achieve a test accuracy of 81.2% on the TensorFlow dataset rock_paper_scissors so there is room for improvement. Here's a short video of predicting from webcam:
![](figure/demo.gif)

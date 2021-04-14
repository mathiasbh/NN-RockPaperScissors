import sys
sys.path.append('src')

from config import *
import cv2 as cv


# Load trained model
PATH = 'D:/Google drive/Coding/MachineLearning/NN-RockPaperScissors/_MODELS/saved_model_googlenet_5'
model = tf.keras.models.load_model(PATH)


def classify_image(model, image):
    prediction = model(np.array([image]))
    prediction_class = np.argmax(prediction)
    prediction_percent = prediction[0][prediction_class]
    prediction_name = IMAGE_CLASS[str(prediction_class)]
    return(prediction_name, prediction_percent)



capture = cv.VideoCapture(0)

while True:
    # Show video capture frame by frame
    ret, frame = capture.read()

    if frame is None:
        break


    # Extract the image from the rectangle
    roi = frame[100:400, 175:475]  # y, x
    img = cv.cvtColor(roi, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img/255


    # Classify
    prediction, pct = classify_image(model, img)

    # Display the resulting frame
    cv.rectangle(frame, (175, 100), (475, 400), (0, 0, 255), 3)
    cv.putText(frame, "{:10s}".format(prediction), (175,80), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)
    cv.putText(frame, "({:3.0f}%)".format(pct*100), (360,80), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)
    cv.imshow('Frame', frame)

    keyboard = cv.waitKey(1)
    if (keyboard == 'q') or (keyboard == 27):
        break


capture.release()
cv.destroyAllWindows()

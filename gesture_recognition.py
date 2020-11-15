import copy
import cv2
import math
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('C:/Users/charl/OneDrive/Desktop/gesture_classifier.h5')
gesture_names = {0: 'Palm',
                 1: 'L',
                 2: 'Fist',
                 3: 'Fist_Moved',
                 4: 'Thumb',
                 5: 'Index',
                 6: 'Ok',
                 7: 'Palm_Moved',
                 8: 'C',
                 9: 'Down'}


def remove_bg(frame):
    bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
    mask = bgModel.apply(frame, learningRate=0)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    return res


def predict_image(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)

    # model.predict() returns an array of probabilities,
    # np.argmax gets the index of the highest probability.
    result = gesture_names[np.argmax(pred_array)]

    # Display only 2 decimal points
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(f'Result: {result}, Score: {score}')
    return result, score


cap = cv2.VideoCapture(0)
cap.set(10, 200)

while True:
    ret, frame = cap.read()

    # Use bilateral smoothing filter instead of kernel
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = cv2.flip(frame, 1)

    # Draw rectangle around ROI
    cv2.rectangle(frame, (int(0.5*frame.shape[1]), 0),
                  (frame.shape[1], int(0.8*frame.shape[0])), (255, 0, 0), 2)

    # Remove background, crop out ROI
    img = remove_bg(frame)
    img = img[0:int(0.8*frame.shape[0]),
              int(0.5*frame.shape[1]): frame.shape[1]]

    # Convert to binary colour image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (41, 41), 0)

    ret, thresh = cv2.threshold(
        blur, 60, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use threshold for predictions
    predict = np.stack((thresh, )*1, axis=-1)
    predict = cv2.resize(predict, (224, 224))
    predict = predict.reshape(1, 224, 224, 1)
    prediction, score = predict_image(predict)

    # Add prediction and score to thresholded image
    cv2.putText(thresh, f"Prediction: {prediction} ({score}%)",
                (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    cv2.imshow("Original", frame)
    # cv2.imshow("Mask", img)
    # cv2.imshow("Blur", blur)
    cv2.imshow("Thresholded", thresh)

    # Press Esc to quit
    k = cv2.waitKey(25) & 0xFF
    if k == 27:
        break

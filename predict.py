import glob
import argparse
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from matplotlib import pyplot as plt


MODEL_FILENAME = 'model'
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=False,
                help="trained model file name (without the .h5 extension)",
                default=MODEL_FILENAME, metavar="filename")
args = vars(ap.parse_args())
MODEL_FILENAME = args["model"]

classifier = load_model(f"{MODEL_FILENAME}.h5")

# same order as files
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
for file in glob.glob("./images/real_life_examples/*"):
    print(file)
    img = cv2.imread(file)
    plt.imshow(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, dsize=(48, 48))
    img = np.expand_dims(img, axis=0)
    prediction = classifier.predict(img)[0]
    label = emotion_labels[prediction.argmax()]
    plt.xlabel(label)
    plt.show()

    print(label)

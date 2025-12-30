import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# =====================
# VARIABLES
# =====================
path = 'dataset'
images = []
classNumber = []

testRatio = 0.2
validationRatio = 0.2


# =====================
# LOAD DATASET
# =====================
# Get sorted list of class folders
myList = sorted(
    [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))],
    key=lambda x: int(x) if x.isdigit() else x
)
numberOfClass = len(myList)
print("Classes:", myList)

# Loop through classes and images
for idx, class_name in enumerate(myList):
    folder = os.path.join(path, class_name)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(cv2.resize(img, (32, 32)))
            classNumber.append(idx)

# Convert to numpy arrays
images = np.array(images)
classNumber = np.array(classNumber)

# =====================
# SPLIT DATA
# =====================
x_train, x_test, y_train, y_test = train_test_split(images, classNumber, test_size=testRatio)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validationRatio)

# =====================
# PREPROCESSING
# =====================
def imgpreprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

x_train = np.array(list(map(imgpreprocessing, x_train)))
x_test = np.array(list(map(imgpreprocessing, x_test)))
x_val = np.array(list(map(imgpreprocessing, x_val)))

x_train = x_train.reshape(-1, 32, 32, 1)
x_test = x_test.reshape(-1, 32, 32, 1)
x_val = x_val.reshape(-1, 32, 32, 1)

y_train = to_categorical(y_train, numberOfClass)
y_test = to_categorical(y_test, numberOfClass)
y_val = to_categorical(y_val, numberOfClass)

# =====================
# DATA AUGMENTATION
# =====================
dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    rotation_range=10
)
dataGen.fit(x_train)

# =====================
# MODEL
# =====================
def myModel():
    model = Sequential()
    model.add(Conv2D(32, (5,5), activation='relu', input_shape=(32,32,1)))
    model.add(Conv2D(32, (5,5), activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(numberOfClass, activation='softmax'))

    model.compile(
        Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = myModel()
model.summary()

# =====================
# TRAINING
# =====================
history = model.fit(
    dataGen.flow(x_train, y_train, batch_size=50),
    steps_per_epoch=len(x_train)//50,
    epochs=15,
    validation_data=(x_val, y_val)
)

# =====================
# EVALUATION
# =====================
score = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", score[1])

# =====================
# SAVE MODEL (CORRECT WAY)
# =====================
model.save("mytrained.keras")
print("Model saved as mytrained.keras")

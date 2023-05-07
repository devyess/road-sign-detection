import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.special import legendre

data = []
labels = []
classes = 43
cur_path = os.getcwd()

# Retrieving the images and their labels 
for i in range(classes):
    path = os.path.join(cur_path, 'train', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(os.path.join(path, a))
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)

# Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Load the test data
test_data = pd.read_csv('Test.csv')
test_labels = test_data["ClassId"].values
test_imgs = test_data["Path"].values

data=[]

for img in test_imgs:
    image = Image.open(os.path.join(cur_path, img))
    image = image.resize((30, 30))
    data.append(np.array(image))

X_test = np.array(data)

# Load the trained model
model = load_model("my_model.h5")

# Predict the labels of test data
pred = model.predict(X_test)

# Convert predictions to class labels
pred_classes = np.argmax(pred, axis=1)

# Convert true labels to one-hot encoding
true_classes = to_categorical(test_labels, 43)

# Generate confusion matrix
cm = confusion_matrix(true_classes.argmax(axis=1), pred_classes)

# Plot confusion matrix
plt.figure(figsize=(20, 20))
sns.heatmap(cm, annot=True, cmap='BuPu')
plt.show()

# Compute Legendre moments of test data
N = 10  # number of moments to compute
M = np.zeros((len(X_test), N))
for i, img in enumerate(X_test):
    for n in range(N):
        pn = legendre(n)
        integral = np.zeros(2)
        for j, axis in enumerate([0, 1]):
            integrand = lambda t: img[:, :, axis] * pn((t+1)/2)
            integral[j] = np.sum(integrand(np.linspace(-1, 1, img.shape[0])) * (2/img.shape[0]))
        M[i, n] = (n+1)/2 * np.sum(integral)

# Plot Legendre moments
plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(np.arange(N), M[i], 'o-')
plt.show()

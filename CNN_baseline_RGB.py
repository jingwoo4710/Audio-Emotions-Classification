# Importing required libraries 
# Keras
import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling2D
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_resnet_v2 import InceptionResNetV2


# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

# Other  
import librosa
import librosa.display
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import seaborn as sns
import glob 
import os
from tqdm import tqdm
import pywt
import cv2 as cvlib
import pickle
import IPython.display as ipd  # To play sound in the notebook

# # Load the data in GoogleDrive 
# from google.colab import drive
# drive.mount('/content/gdrive')

# # Kaggle.json file location
# import os
# os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/My Drive/Kaggle"

# # Change the working directory
# %cd /content/gdrive/My Drive/Kaggle


# Load RGB sound file
X = np.load('RGB_sound.npy')
y = np.load('y_label_hot_sound.npy')

# New Model is adopted since GPU limit and data size
model = Sequential()
model.add(Conv2D(128, 8, padding='same', input_shape=(299, 299, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(8)))
model.add(Conv2D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(14)) # Target class number
model.add(Activation('softmax'))

opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)

# Model compile
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
model_history=model.fit(X, y, batch_size=32, epochs=50)

# Save model and weights
model_name = 'Emotion_Transfer_Model_RGB.h5'
save_dir = os.path.join(os.getcwd(), 'RGB_saved_models')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Save model and weights at %s ' % model_path)

# Save the model to disk
model_json = model.to_json()
with open("tr_model_json_RGB.json", "w") as json_file:
    json_file.write(model_json)

# loading json and model architecture 
json_file = open('tr_model_json_RGB.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("RGB_saved_models/Emotion_Transfer_Model_RGB.h5")
print("Loaded model from disk")
 
# Keras optimiser
opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# CV
kf = KFold(5, random_state = 1024)

all_scores = []

for train_idx, test_idx in kf.split(X):
  X_train, X_test = X[train_idx], X[test_idx]
  y_train, y_test = y[train_idx], y[test_idx]

  score = loaded_model.evaluate(X_test, y_test, batch_size = 32)
  all_scores.append(score)

all_scores = np.asarray(all_scores)

loss, acc = np.mean(all_scores, axis = 0)

print("%s: %.2f%%" % (loaded_model.metrics_names[1], acc*100))
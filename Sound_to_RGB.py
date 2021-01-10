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

# Load Data
df = pd.read_csv('Data_path.csv')

# GPU 문제 -> 데이터 축소
df = df.loc[df['source'] == 'RAVDESS']


def norm (X):
  # Data normalization 
  mean = np.mean(X, axis=0)
  std = np.std(X, axis=0)

  return (X - mean)/std
  

# 40 - 50 mins 
n_row, n_cols = df.shape
n_class = len(df.labels.unique())

# RGB sound
RGB_sound = np.zeros((n_row, 299, 299, 3))
y_label_sound=np.zeros((n_row, 1))
y_label_hot_sound=np.zeros((n_row, n_class))

n_fft = 1024 # frame length
hop_length = 512

for i in range(n_row) :

    print('loop nb', i)
    print('emotions', df['labels'].iloc[i])
    filepath = df['path'].iloc[i]

    # open the audio file
    clipnoise, sample_rate = librosa.load(filepath, sr=22500)


    scales = np.arange(1, 128)
    waveletname = 'morl'

    coeffnoise, freqnoise = pywt.cwt(clipnoise, scales, waveletname)
    scalogramimg=cvlib.resize(coeffnoise, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)


    stft = librosa.stft(clipnoise, n_fft=n_fft, hop_length=hop_length)
    stft_magnitude, stft_phase = librosa.magphase(stft)
    stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
    


    spectrogramimg=cvlib.resize(stft_magnitude_db, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)


    # mfcc
    mfcc = librosa.feature.mfcc(y=clipnoise, sr=sample_rate, n_mfcc=200)


    mfccimg=cvlib.resize(mfcc, dsize=(299, 299), interpolation=cvlib.INTER_CUBIC)


    # RGB
    RGB_sound[i, :, :, 0] = spectrogramimg
    RGB_sound[i, :, :, 1] = scalogramimg
    RGB_sound[i, :, :, 2] = mfccimg



    print('finish loop nb', i)

# Normalization
RGB_sound[:, :, :, 0] = norm(RGB_sound[:, :, :, 0])
RGB_sound[:, :, :, 1] = norm(RGB_sound[:, :, :, 1])
RGB_sound[:, :, :, 2] = norm(RGB_sound[:, :, :, 2])


# One hot encode the target 
lb = LabelEncoder()
y = np_utils.to_categorical(lb.fit_transform(df['labels']))


print(lb.classes_)


# Pickel the lb object for future use 
filename = 'labels'
outfile = open(filename, 'wb')
pickle.dump(lb, outfile)
outfile.close()

np.save('RGB_sound', RGB_sound)
np.save('y_label_hot_sound', y)

print('saved to disk')
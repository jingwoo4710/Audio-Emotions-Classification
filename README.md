![Generic badge](https://img.shields.io/badge/version-1.1.1-green.svg)
# Compare performances of CNN with that of Transfer learning
There are tons of examples using CNN model to classify emotions from the audio. However, very limited examples can be found using transfer learning. This causes a question  "why people are more likely to use CNN models than other models?". From the generalization point of view, the well-knonw pre-trained models are more applicable for emotion detections such as [Wavenet](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio), and [InceptionResNetV2](https://arxiv.org/abs/1602.07261). In this project, the comparison of performances of CNN model and pre-trained InceptionResNetV2 is made. 
# Models
## 1. CNN 
In the Kaggle, there are many examples of CNN model to classify emotions from the sound. [The best CNN model](https://www.kaggle.com/ejlok1/audio-emotion-part-3-baseline-model) is adopted for a base-line model. Therefore, this model is heavily borrowed from the Kaggle notebook. 
## 2. InceptionResNetV2
With the method of transfer leanring, InceptionResNetV2 model is used as a feature extractor. In order to use InceptionResNetV2, the default data size is (299, 299, 3). Therefore, the sound file is converted to RGB picture, red being spectrogram, green being scalogram and blue MFCC(Mel-Frequency Cepstral Frequencys). 

# Dataset
Dataset is obatained from the [Kaggle](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio). RAVDESS dataset is speech audio-only files. The RAVDESS contains 24 professional actors(12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech emotions includes calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression.

Example
------------------------------------
1. Gender - Female, Emotion - Angry 
<div>
  <img src = "https://user-images.githubusercontent.com/70493869/104118197-88b2f380-536a-11eb-9a3e-b0d0c01d700f.png"></img>
  <img src = "https://user-images.githubusercontent.com/70493869/104120027-109efa80-5377-11eb-9c83-3c7f9d84421a.png"></img>
</div>

2. Gender - Male, Emotion - Angry 
<div>
  <img src = "https://user-images.githubusercontent.com/70493869/104120065-5491ff80-5377-11eb-8097-bfef1573ae99.png"></img>
  <img src = "https://user-images.githubusercontent.com/70493869/104120066-552a9600-5377-11eb-8239-440cd642df0d.png"></img>
</div>


# Result

## 1. CNN Model

```
              precision    recall  f1-score   support

       angry       0.55      0.67      0.60      1465
     disgust       0.57      0.44      0.50      1413
        fear       0.46      0.48      0.47      1393
       happy       0.43      0.57      0.49      1467
     neutral       0.59      0.52      0.55      1429
         sad       0.58      0.48      0.53      1473
    surprise       0.83      0.69      0.75       482

    accuracy                           0.54      9122
   macro avg       0.57      0.55      0.56      9122
weighted avg       0.55      0.54      0.54      9122
```
<div>
  <img src = "https://user-images.githubusercontent.com/70493869/104126335-448e1600-539f-11eb-9132-038b57894018.png"></img>
</div>



## 2. InceptionResNetV2







# Installation
All the codes are based on Colab Notebooks, using GPU on the notebook settings. Also, `requirements.txt` file is attached. 

![Generic badge](https://img.shields.io/badge/version-1.1.1-green.svg)
# Performances of CNN with that of Transfer learning on Audio Emotions Classification
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

|Model|Accuracy|Precision|Recall|F1|
|------|---|---|---|---|
|CNN|0.42|0.45|0.41|0.42|
|INCEPTION|0.47|0.49|0.45|0.43|


## 1. CNN Model
<div>
  <img src = "https://user-images.githubusercontent.com/70493869/104155161-e6f2db80-5429-11eb-94a0-39fc2db93afa.png"></img>
</div>



## 2. InceptionResNetV2
<div>
  <img src = "https://user-images.githubusercontent.com/70493869/104155048-af842f00-5429-11eb-9e8e-1c304958e59c.png"></img>
</div>






# Installation
All the results are based on Colab Notebooks, using GPU on the notebook settings. Also, `requirements.txt` file is attached. 


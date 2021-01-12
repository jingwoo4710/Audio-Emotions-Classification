![Generic badge](https://img.shields.io/badge/version-1.1.1-green.svg)
# Performances of CNN with that of Transfer learning on Audio Emotions Classification
There are tons of examples using the CNN model to classify emotions from the audio. However,  there are a few examples of using transfer learning. This causes the question of why people are more likely to use CNN models than other models? From the generalization point of view, the well-known pre-trained models are more applicable for emotion detections such as [Wavenet](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio), and [InceptionResNetV2](https://arxiv.org/abs/1602.07261). In this project, the comparison of performances of the CNN model and pre-trained InceptionResNetV2 is made.


# Installation
All the results are based on Colab Notebooks, using GPU on the notebook settings. Also, the `requirements.txt` file is attached. 


# Models
## 1. CNN 
In the Kaggle, there are many examples of the CNN model to classify emotions from the sound. [The best CNN model](https://www.kaggle.com/ejlok1/audio-emotion-part-3-baseline-model) behaves as a base-line model. Therefore, this base-line model is heavily borrowed from the Kaggle notebook.
## 2. InceptionResNetV2
With the method of transfer learning, the InceptionResNetV2 model acts as a feature extractor. In order to use InceptionResNetV2, the default data size is (299, 299, 3). Therefore, the sound file is converted to an RGB picture, red being spectrogram, green being scalogram, and blue MFCC(Mel-frequency Cepstral Frequency). 

# Dataset
Dataset is obtained from the [Kaggle](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio). RAVDESS dataset is speech audio-only files. The RAVDESS contains 24 professional actors(12 female, 12 male), vocalizing emotions including calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression.

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



# Conclusion
Since there is a RAM limit in Colab notebooks, a limited Dataset is used to train InceptionResNetV2. Even though this problem, the performance is slightly higher than that of the CNN model. To be a fair comparison, every other control variables should be controlled. In this case, a direct comparison of performances is not valid. Therefore, a new CNN model designed to be trained by the RGB Sound dataset is introduced. This model is simple to help see the differences between the two models. 


|Model|Accuracy|Precision|Recall|F1|
|------|---|---|---|---|
|CNN(RGB Audio)|0.06|0.004|0.07|0.09|

The table above tells the performance is far less than expected. Since the dataset has a good quality audio file, it's straightforward to tell the differences between emotions. However, the RGB version of the Audio file needs more steps to differentiate emotions. In other words, InceptionResNetV2 as a feature extractor is better than just the CNN model. Because the performances of InceptionResNetV2 is already proved, this could be taken for granted.

To summarize, the transfer learning method indeed shows a better performance than the CNN model. This might be because InceptionResNetV2 has a better feature extractor. But, InceptionResNetV2 consumes a lot of RAM in the Colab notebook. Hence, for the people aiming for better performances, the transfer learning method is recommended. In contrast, a simple CNN model could work perfectly with very limited use of RAM. Therefore, for the people aiming for a time, and efficiency, the CNN model is recommended. 




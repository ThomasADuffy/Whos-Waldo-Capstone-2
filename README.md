# **Who actually is Waldo?** <!-- omit in toc -->
### Creating an Image Classification CNN to Predict if it is Waldo<!-- omit in toc -->  

![Title pic][title_pic]

[title_pic]:/images/misc_imgs/Title-Banner.jpg

# **Table of Contents** <!-- omit in toc -->
- [**Introduction**](#introduction)
- [**EDA and Finding Data**](#eda-and-finding-data)
- [**Readme Images and Data Credits/Sources**](#readme-images-and-data-creditssources)
  - [Readme Images sources](#readme-images-sources)
  - [Datasets sources](#datasets-sources)

# **Introduction**
Ever since I was a kid, I was terrible at where’s waldo . It has been a bugging issue until today where I can conquer over Waldo with machine learning and convoluted neural networks. On a more serious note, everybody has heard of or has played Where’s Waldo?(also known as Wheres Wally?). Being very interested in Neural Networks, I wanted to create a Convolutional Neural Network(CNN) which is able to tell if an image is Waldo or not. I also used Tensorflow and Keras to create the CNN and also utilized my GPU to run my CNN which made training magnitudes faster than training on a CPU. Other libraries I utilized was Matplotlib, Scikit-learn, Scikit-Image, and some other basic libraries. I created the code to be able to run on any computer which downloads the scripts due to utilizing the os, sys libraries and making import/export directories universal per computer. This ended up in three main classes I utilized and three jupiter notebooks which were used for EDA, testing, and plotting.
![Tensorflow_Keras][Tensorfow_keras]

[Tensorfow_keras]:/images/misc_imgs/Tensorflow_keras.jpeg  
# **EDA and Finding Data**
The first step of this project was to find data. Now my data would be two classifications of pictures, One of waldo pictures and one of non-waldo pictures. Luckily enough, I found a dataset on a github repository of already pre classed waldo pictures [here](https://github.com/vc1492a/Hey-Waldo). This dataset contains 3 different dimension's of photos: 256 x 256 pixels (317 images|286 of not waldo and 31 of waldo), 128 x 128 pixels (1344 images|1317 of not-waldo and 27 of waldo) , and 64 x 64 pixels (5376 images|5337 of not-waldo and 39 of waldo). It also contains some original images on where these pictures were cropped from. For each of the different dimensions, it had copies in grayscale and black and white. Below is an image representation of the directory structure.  
![Githubwaldo Structure][githubwaldo_structure]

[githubwaldo_structure]:/images/misc_imgs/githubwaldo_structure.jpg
I chose to go with RGB photos because color is a distinctive feature surrounding waldo. Especially when it comes to his iconic red and white t-shirt and hat, in addition his skin color is something that is distinct also.The next choice I made after experimenting with the data was choosing the dimension size. I ended up going with the 64x64 images due to the sheer number of images and also because I wanted to focus my training on waldos face which is a relatively small portion of a page. After looking into the pictures I realized that this dataset was not the best it had a huge class imbalance regardless of the dimensions and when I looked into the pictures classified as waldo, over half of them were not waldo and something very arbitrary as seen below.  
![Bad Waldo1](/images/misc_imgs/badwaldo_1.jpg) ![Bad Waldo2](/images/misc_imgs/badwaldo_2.jpg) ![Bad Waldo3](/images/misc_imgs/badwaldo_3.jpg)
<!-- <img src="https://github.com/ThomasADuffy/Crypto-Capstone-1/blob/master/imgs/ETH_logo.png" height="256">

><span style=font-size:1.1em;>Is the correlation between the count of tweets per day and price of each coin?</span>  
> And if there is a correlation, is this a negative or positive correlation? -->


# **Readme Images and Data Credits/Sources**
## Readme Images sources
Title picture: https://i.ytimg.com/vi/U9g6cHcTkaA/maxresdefault.jpg  
Tensorflow/Keras picture: https://hackernoon.com/tf-serving-keras-mobilenetv2-c167b4b2bb25  
## Datasets sources
Constantinou , Valentino : vc1492a, 2018, Hey Waldo, V1.8, Github, https://github.com/vc1492a/Hey-Waldo  
Waldo Picture Gallery : https://imgur.com/gallery/8exqx  
Waldo Pictures Gallery : https://www.deviantart.com/where-is-waldo-wally/gallery/all


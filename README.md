# **Who actually is Waldo?** <!-- omit in toc -->
### Creating an Image Classification CNN to Predict if it is Waldo<!-- omit in toc -->  

![Title pic][title_pic]

[title_pic]:/images/misc_imgs/Title-Banner.jpg

# **Table of Contents** <!-- omit in toc -->
- [**Introduction**](#introduction)
- [**EDA and Finding Data**](#eda-and-finding-data)
  - [**Testing the WaldoGenerator**](#testing-the-waldogenerator)
- [**Building the WaldoCNN**](#building-the-waldocnn)
  - [**The First Model **](#the-first-model)
  - [**The Second Model**](#the-second-model)
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
Therefor, I had to physically play Where's Waldo? for hours to get a good data set, from where I cropped the images to be 64x64(I did this from these two image galleries [here](https://imgur.com/gallery/8exqx) and [here](https://www.deviantart.com/where-is-waldo-wally/gallery/all)). I ended up with 58 total photos of just waldos face, a few examples are below.  
 ![Good Waldo1](/images/misc_imgs/goodwaldo_1.jpg) ![Good Waldo2](/images/misc_imgs/goodwaldo_2.jpg) ![Good Waldo3](/images/misc_imgs/goodwaldo_3.jpg) ![Good Waldo4](/images/misc_imgs/goodwaldo_4.jpg)

 I still had a major class imbalance, I had 5337 non-waldo pictures and only 58 waldo pictures. Therefor I utilized Tensorflow's Keras ImageDataGenerator to generate more images of waldo which were augmented. One key factor was figuring about the hyperparameters of this image generator. As you can see from above in all the images, his head is always visible and also he is always upright with very little rotation, but he can face either left or right. Within my EDA notebook located [here](https://github.com/ThomasADuffy/Whos-Waldo-Capstone-2/blob/master/notebooks/EDA_ImgGeneration.ipynb) I tested out the images to see what they looked like before and after augmentation so that I knew my parameters were right. After much trial and error I found good parameters and ended up with the ImageGenerator as below:  
 ![Img Gen](/images/misc_imgs/imagegenerator_code.jpg)
 | **Before**  | **After** |
| ------------- | ------------- |
| ![waldob1](/images/misc_imgs/waldobefore1.jpg)  | ![waldoa1](/images/misc_imgs/waldoafter1.jpg)  |
| ![waldob2](/images/misc_imgs/waldobefore2.jpg)  | ![waldoa2](/images/misc_imgs/waldoafter2.jpg)  |
 As you can see I only really shifted him around and changed brightness and zoom with horizontal flipping. In order to prevent data leakage, I split my images into subsets of test and train before generating more images, 10 test and 48 train. Then I did the augmentation on them which I created enough to total the number of non-waldo pictures so I wouldnt have a class imbalance. This ended up being 3952 training pictures and 990 test testing pictures, which when combined with the originals equaled 5000. That was relatively close enough to the total non-waldo pictures so I then split them accordingly to a testing dataset folder of ~1000 both waldo and non-waldo pictures each and did the same for my training dataset folder which contained about ~4000 each. I did this all by creating a class called WaldoGenerator that would generate these picture. To read more about my class click [here](https://github.com/ThomasADuffy/Whos-Waldo-Capstone-2/blob/master/src/waldo_generator.py).  
## **Testing the WaldoGenerator**
I ended up experimenting with the unittest library and creating a script that would test my waldo image generator. What it would do would test my class by instantiating a class with a input file directory with a test image in it and output directory located in my test directory and generate one image. then it would test that exactly one image was generated then delete the image after so you can run the test again seamlessly. The code is as follows:  
![test](/images/misc_imgs/test_code.jpg)  
you can check out the code [here](https://github.com/ThomasADuffy/Whos-Waldo-Capstone-2/blob/master/test/test_waldo_generator.py).  

# **Building the WaldoCNN**
I started building a class which incorporated a CNN as one of the methods and attribute so I could pull various metrics and also load a previous model and utilize the class still if I needed. It also allowed me to save the model seamlessly and also save the metrics as a csv so I can plot with them much easier. I created the CNN using Keras's sequential model and tried two different models.

## **The First Model **
The first model is shown below:  
<img src="/images/plots_structures/Model_v1.jpg" width="450"> <img src="/images/plots_structures/model_V1_FC.jpg" width="350">
<!-- ![test](/images/plots_structures/model_V1_FC.jpg)![test](/images/plots_structures/Model_v1.jpg) -->

## **The Second Model**
The second model is shown below:  
<img src="/images/plots_structures/Model_v2.jpg" width="450"> <img src="/images/plots_structures/model_V2_FC.jpg" width="350">


# **Readme Images and Data Credits/Sources**
## Readme Images sources
Title picture: https://i.ytimg.com/vi/U9g6cHcTkaA/maxresdefault.jpg  
Tensorflow/Keras picture: https://hackernoon.com/tf-serving-keras-mobilenetv2-c167b4b2bb25  
## Datasets sources
Constantinou , Valentino : vc1492a, 2018, Hey Waldo, V1.8, Github, https://github.com/vc1492a/Hey-Waldo  
Waldo Picture Gallery : https://imgur.com/gallery/8exqx  
Waldo Pictures Gallery : https://www.deviantart.com/where-is-waldo-wally/gallery/all


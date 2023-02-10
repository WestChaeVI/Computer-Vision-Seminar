# Data Network Analysis(DNA)
Computer Vision

> 2022.01.19 ~ ing
> I am active as a member of the club of the undergraduate major of the Department of Data Science at the Suwon University.   
   
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Week 1 - Neural Network
+ Structure of a basic ANN(Artificial Neural Network) Forward/Back Propagation
+ CNN(Convolution Neural Network) [references](http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture05.pdf)
+ Conv Layer (Channel, kernel_size, activation, padding, stride, kernel_init, pooling)
+ FC Layer (flatten, Dense, Dropout, pooling)
+ VggNet(Vgg16, Vgg19)
+ Gradient Loss & Overfitting

#### 과제1. Modifying the Image Classification Model In Kaggle
 [과제1을 수행한 Kaggle Link](https://www.kaggle.com/code/westchaevi/beg-tut-intel-image-classification-93-76-accur/edit)   
 Seeking ways to improve gradient loss and overfitting problems by referring to existing Intel ImageNet Challenge codes in Kaggle

> 과제1 Review
> 1. How to efficiently stack layer? => Increase the number of channels + Reduce size loss using padding + Max Pooling size should not be too large.
> 2. Gradient Loss => BatchNormalization + Weight_init + activaiton='Relu'
> 3. Overfitting Problem => Dense, Dropout
> 4. Merit of Global Average Pooling => Reduces the number of parameters + Prevents loss of spatial information
> 5. Batchsize & Learning rate are positively correlated.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Week 2 - Transfer Learning
+ Imagenet Chanllenge Review
+ Pre-trained & Fine tuning
+ 3 Strategies for Fine tuning
+ Feature Extraction from Pretrained model
+ Lower layers of the model extract 'local' and very 'general' feature maps(edges, colors, textures, etc.). On the other hand, the Higher layer (dog's eyes or cat's ears) to extract more abstract feature.  [Transfer Learning에 대한 블로그 포스팅](https://westchaevi.tistory.com/3)

#### 과제2. Manage Image Directory
[과제2를 수행한 코드](https://github.com/WestChaeVI/Date-Network-Analysis/blob/main/CV_seminar_project_week_2_%EA%B3%BC%EC%A0%9C.ipynb)
![0204232130857502](https://user-images.githubusercontent.com/104747868/216772710-37c7c120-ecfb-4059-8ad4-7cf115eb9027.jpg)


It took 1312 pictures of dolphins, sharks, and whales, created a class according to the name of the fish, and created a function to put the image in the corresponding class in a 7(train) : 2(valid) : 1(test) ratio.

> 과제2 Review
> 1. It was an opportunity to experience firsthand how important it is to make code efficiently and simply.
> 2. Reduce the length of code by using dictionaries.
> 3. This assignment was able to optimize the function through enough internet search, but I wrote code to learn about handling Class and self methods.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Week 3 - ImageNet Challenge 2014
+ Making py module
+ How to visualize images using libraries (cv2_imshow, plt.imshow, PIL.open)
> Since plt.imshow reads the image backwards as BGR rather than RGB when it is input, there is one thing that needs to be changed.
> plt.imshow(img[:,:,::-1])
+ Reading and Reviewing [VggNet](https://arxiv.org/pdf/1409.1556.pdf) paper
> 1. Create the most efficient model -> Layer deep + no loss of gradient
> 2. It was stacked deep using a 3x3 filter.
> 3. The 1x1 filter made sense


#### 과제3 - (1) Visualizing different images for each class for each column.
#### 과제3 - (2) Visualizing by changing image pixel values

[과제3을 수행한 코드](https://github.com/WestChaeVI/Date-Network-Analysis/blob/main/CV_seminar_project_week_3_%EA%B3%BC%EC%A0%9C.ipynb)
![0204013449759406](https://user-images.githubusercontent.com/104747868/216657163-7755fa5f-ae37-40dc-90bd-db3e43c9bd30.jpg)

> (1) Visualize different images for each row of images of dolphins, sharks, and whales (Condition: Use only one 'for statement')

> (2) Load any dolphin image and change the pixels of the image to add color.


> 과제3 Review
> 1. I was able to study how to import an image and visualize it in the desired rows and columns.
> 2. Using dictionaries and enumerate functions can effectively shorten the length of your code.
> 3. The plt.tight_layout() function was used to prevent the images from overlapping.
> 4. I was able to study how to handle pixel values of images. ~~I wanted to use the cv2.line() function intensely..~~

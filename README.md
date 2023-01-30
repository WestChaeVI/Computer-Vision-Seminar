# Data Network Analysis(DNA)
Computer Vision

> 2022.01.19 ~ ing
> I am active as a member of the club of the undergraduate major of the Department of Data Science at the University of Suwon.   
   
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

# Data Network Analysis(DNA)
Computer Vision

> 2023.01.19 ~ 2023.03.09

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
 
 ![0223221548355065](https://user-images.githubusercontent.com/104747868/220949644-4c0ff120-7be5-4b5d-bde0-93217d215e4b.jpg)
 
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

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Week 4 - Augmentation in Pytorch
+ Learn how Pytorch works.
> 1. Preparing DataSet
> 2. Creating DataClass
> 3. Creating Model
> 4. Learning

+ Create a Pytorch dataset class in Colab

+ Image augmentation
> 1. A.Resize(224,224)
> 2. A.Transpose(p=0.5)
> 3. A.HorizontalFlip(p=0.5)
> 4. A.VerticalFlip(p=0.5)
> 5. A.ShiftScaleRotate(p=0.5)
> 6. A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5)
> 7. A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5)
> 8. A.ChannelShuffle()
> 9. A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
> 10. A.CoarseDropout(p=0.5)
> 11. ToTensorV2()


[Customizing Pytorch Dataset Class](https://github.com/WestChaeVI/Data-Network-Analysis/blob/main/week4_Pytorch_Dataset_Class_%EC%BB%A4%EC%8A%A4%ED%85%80.ipynb)

#### 과제4. Visualizing the original image and the augmented image
[과제4를 수행한 코드](https://github.com/WestChaeVI/Data-Network-Analysis/blob/main/CV_seminar_week4_%EA%B3%BC%EC%A0%9C.ipynb)
![218383559-5690ebf5-28a9-43ac-af64-c17748536e9f](https://user-images.githubusercontent.com/104747868/218384138-017b75bb-8d92-414b-9688-e438b2872b6c.png)


> 과제4 Review

> To visualize an image, we need to create the inverse function of Augmentations.
> 1. Change the channel axis using the permute module. **(3, 224, 224) -> (224, 224, 3)**
> 2. Convert from tensor to numpy
> 3. renormalizing, and astype(int)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Week 5 - GoogLeNet & ResNet

### [GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf) = 1x1 Conv + Concatenate + Auxiliary classifier
![0223233526203321](https://user-images.githubusercontent.com/104747868/220949449-67914dd9-7241-44e8-8c2e-f0347b492021.jpg)

##### Focusing on Depth and **Width**
> Using filters of various sizes.
##### Considering Computational Complexity 
> It makes it possible to use 1x1 filter efficiently.
> It was able to layer deeper.
##### Merits of '1x1 Conv'
> 1. It can handle the channel axis.
> 2. It can give nonlinearity.
##### Naïve Version vs **Dimension Reduction Version**(adopted)
> 1. Naïve Version = previous layer + **5x5 conv** + filter concatenation
> 2. Dimension Reduction Version = previous layer + **1x1 conv + 5x5 conv** + filter concatenation


### [ResNet](https://arxiv.org/pdf/1512.03385.pdf) = Plain net + Identity Mapping
![0224001643651415](https://user-images.githubusercontent.com/104747868/220948683-fbc046eb-9967-4536-8a71-23199dcf1cad.jpg)

##### "If it stack 1000 layers without Overfitting and Gradient Vanishing, it'll get 100% accuracy, right?
---> "Is learning better network as easy as stacking more layer?"

---> ResNet turned the focus differently from the previous State of the Art(Sota) models.

##### The layers were stacked deeper, but the performance was not better than the shallow ones.
---> It is not problem of Overfitting, Gradient Vanishing

---> Degradation Problem (of training accuracy)

##### Identity Mapping
> 1. H(x) = f(x) + x  (x is Input, f(x) is Conv)
> 2. Definition of ploblem : H(x) - x = 0 
> 3. Let's adjust the optimization well to zero the residuals.

##### propagation & back propagation
> Due to the nature of the conv layer, as the layer deepens, the image is viewed more globally and the high-level feature is learned.

> Now, using reserved block, each layer learns H(x) from the value received from identity mapping, so learning/convergence is easier because their mission is clearer.

> Since shortcut is simply a plus node, the complexity does not increase, and the gradient can be propagated even in backpropagation, solving the Gradient Vanishing problem.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Week 6 - ResNet50 Image Classification 1 

[Pytorch Code](https://github.com/WestChaeVI/Data-Network-Analysis/blob/main/Task/CV_seminar_project_week_6_%EA%B3%BC%EC%A0%9C.ipynb)

+ The code was written in pytorch.
> 1. Preparing DataSet(train,valid)
> 
> 2. Model is the ResNet50(pretrained=True) and Fine Tuned fc layer output is '3'(dolphin,shark,whale)
> 
> 3. Using CE Loss ( nn.CrossEntropyLoss() )
> 
> 4. Optimizer is 'Adam'

+ The Best Acc and Epoch (valid)
> Acc : **90.87%** ,  Epoch : **38 / 50**

### Acc & Loss Ploting (epochs : 1~50)
![image](https://user-images.githubusercontent.com/104747868/222973434-7010b3e4-2046-402a-82ab-8af2b3ce8a55.png)

### Ploting Prediction for Random Image
![image](https://user-images.githubusercontent.com/104747868/222973547-db96579b-77d9-4fb7-9a42-0db2b9e3a42e.png)
+ **0 : dolphin, 1 : shark, 2 : whale**
> Among the learned weights, I take the weights of the best accuracy epoch, bring the model back up, and visualize making predictions for random photograph.

> If you look at the picture on the rightmost side of the above picture, you can see that the model learned the picture where the answer is a **dolphin** and incorrectly predicted it as a **"whale"**.


> Plotting just below the image is the probability that the model predicts for each label. As you can see in the picture, you can see that they are similar.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Week 7 - ResNet50 Image Classification 2 

The learning weight when it was the best epoch was taken and applied to the test set to conduct model evaluation.

[Pytorch Code](https://github.com/WestChaeVI/Data-Network-Analysis/blob/main/Task/CV_seminar_project_week_7(testing).ipynb)

+ ResNet50 train 100 epochs
> The Best Acc and Epoch (valid)   
> Acc : **91.93%** ,  Epoch : **56 / 100**   

The model weights for epoch 56 were stored, and the stored weights were applied when the model was re-declared.   

+ testset evaluate (epochs : 30)
> The Best Acc and Epoch (test)   
> Acc : **90.77%** ,  Epoch : **15 / 30**


### Acc & Loss Ploting (epoch : 1~30)
![image](https://user-images.githubusercontent.com/104747868/224246210-02a7319e-95d1-4912-887d-6bebbbf89684.png)

+ Conclusion
> At the 15th epoch, 90.769% was achieved based on the test set.

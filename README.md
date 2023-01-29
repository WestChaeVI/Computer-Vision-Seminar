# My-First-Project

## week 4 - GAN 2
+ pix2pix - [Image-to-Image Translation with Conditional Adversarial Networks(2017)](https://arxiv.org/pdf/1611.07004.pdf) 논문 리뷰
+ WGAN ( Wasserstein loss, Lipshitz 제약 ) 개념   
+ Cycle GAN ( 여러개의 loss를 이용 ) 개념 & 논문 리뷰 & 실습   
+ Neural Style Transfer - [Image Style Transfer Using Convolutional Neural Networks(2015)](https://arxiv.org/pdf/1508.06576.pdf) 논문 리뷰 & 실습 [code](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/GAN/Style_transfer_20210802.ipynb)   
+ SR GAN - [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network(2017)](https://arxiv.org/pdf/1609.04802.pdf) 논문 리뷰 & 실습[code](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/GAN/SRGAN_20210615.ipynb)   
#### 개인 프로젝트 1 - fine tuning으로 style transfer의 style을 더욱 강력하게 입혀보기
> vggnet을 fine tuning하여 심슨 이미지를 분류하는 모델을 만든 뒤, 해당 모델의 ConV Network을 이용하여 좀 더 강한 심슨풍의 이미지를 유도해봤습니다!   
> [VggNet fine tuning 적용 코드](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/GAN/project1/pretraining_style_transfer.ipynb)   
> [style transfer 적용 코드](https://github.com/inhovation97/Get-an-education-Computer-Vision/blob/main/GAN/project1/style_transfer_in_pytorch.ipynb)   
>   
> ![image](https://user-images.githubusercontent.com/59557720/161210449-88875252-8fbd-446c-ab45-3e27a79e5024.png)
>   
> 보다시피 fine tuning 하면 확실히 style이 강하게 입혀집니다.   
> 우연치 않게 content image와 style image 모두 벽에 액자가 걸려있었는데, 같은 객체로 인식되어 만화 그대로 액자가 입혀져 아주 만족스러웠습니다.   
>    
> 여러개의 이미지를 추론했는데, 머리 있는 사람은 캐릭터로 인식하지 못한 반면 머리가 없는 주호민은 캐릭터로 인식하여 노란색이 입혀짐!   
>    
> 오히려 finetuning 과정에서 심슨 풍의 이미지를 과적합시키고 싶었지만, 큰 데이터셋이나 개발 환경 등이 너무 아쉬웠음.
> 

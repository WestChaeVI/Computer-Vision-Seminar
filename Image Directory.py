import os
import glob
import cv2
import time

class Make_dataset_dir():
  def __init__(self, root_dir):
    self.root_path = root_dir+'/' if root_dir[-1] != '/' else root_dir # root_dir = '/content/drive/MyDrive/CV_seminar_project/'
    self.img_path_list = root_dir+'original' # Parent path of passed images
    self.trainset_path = root_dir+'train/'
    self.validset_path = root_dir+'valid/'
    self.testset_path = root_dir+'test/'
    self.class_list = ['dolphin', 'shark', 'whale']

  # Estimate Split
  def est_sp(self):
    dolphin_img_list = glob.glob('/content/drive/MyDrive/CV_seminar_project/original/dolphin/*')
    shark_img_list = glob.glob('/content/drive/MyDrive/CV_seminar_project/original/shark/*')
    whale_img_list = glob.glob('/content/drive/MyDrive/CV_seminar_project/original/whale/*')
    
    dic = {'dolphin':dolphin_img_list, 'shark': shark_img_list, 'whale': whale_img_list} # 딕셔너리 생성
    for key in dic.keys():
    print(f'{key} 이미지가 ', len(dic[key]), '개 있습니다.')
  print('------------------------------------------------------------------------------')

    length_list = []
    for key in dic.keys():
      print(f'{key} 이미지는 train, valid, test셋에 대해 ', int(len(dic[key])*0.7), int(len(dic[key])*0.2), int(len(dic[key])*0.1)    

  # Make Directory
  def mk_dir(self):
    '''train, valid, test 폴더를 만들고, 내부에는 클래스 별 폴더를 추가로 만들어 주세요.'''
    '''Create train, valid, test folders, and create additional folders for each class inside.'''

    dataset_dir_list = [self.trainset_path, self.validset_path, self.testset_path]
    for dataset_dir in dataset_dir_list:
      for cls in self.class_list:
        os.makedirs(dataset_dir+cls, exist_ok=True)
    print('디렉토리 생성을 완료하였습니다.')
    
  def move_img(self):
    '''mk_dir에서 만든 폴더들에 각 클래스에 맞는 이미지를 배당해주세요. train, valid, test에 각각 7: 2: 1'''
    '''Allocate images for each class to the folders created in mk_dir. 7: 2: 1 for train, valid, and test respectively'''

    dolphin_img_list = glob.glob('/content/drive/MyDrive/CV_seminar_project/original/dolphin/*')
    shark_img_list = glob.glob('/content/drive/MyDrive/CV_seminar_project/original/shark/*')
    whale_img_list = glob.glob('/content/drive/MyDrive/CV_seminar_project/original/whale/*')
    
    dic = {'dolphin':dolphin_img_list, 'shark': shark_img_list, 'whale': whale_img_list} # 딕셔너리 생성
    length_list = []
    for key in dic.keys():
      length_list.append([int(len(dic[key])*0.7), int(len(dic[key])*0.2), int(len(dic[key])*0.1)]) # 클래스 별 분할 개수 리스트 생성

    for i,key in enumerate(dic.keys()):  # Use enumerate function
      spliting_length = length_list[i]

      for ii, img_path in enumerate(dic[key]):
        if ii+1 <= spliting_length[0] : # If the number of train set is,
          img = cv2.imread(img_path)
          img_name = img_path.split('/')[-1]
          cv2.imwrite(self.trainset_path + '/' + key + '/' + img_name, img)

        elif spliting_length[0] < ii+1 and ii+1 <= spliting_length[0] + spliting_length[1]: # If the number of valid set is,
          img = cv2.imread(img_path)
          img_name = img_path.split('/')[-1]
          cv2.imwrite(self.validset_path + '/' + key + '/' + img_name, img)

        else:
          img = cv2.imread(img_path)
          img_name = img_path.split('/')[-1]
          cv2.imwrite(self.testset_path + '/' + key + '/' + img_name, img)
    print('데이터 스플릿이 전부 완료되었습니다.') # All data splits are complete.

  def run(self):  # Using time library to print total time spent
    start = time.time()
    self.mk_dir()
    self.move_img()
    print('총 소요시간: ', time.time()-start)

  def checking_dirs(self):
    path_list = [self.trainset_path, self.validset_path, self.testset_path]

    for i,path in enumerate(path_list):
      length_dic = {}
      for cls in self.class_list:
        length_dic[cls] = len(glob.glob(path+cls+'/*'))

      if i==0:
        for key in length_dic:
          print( f'trainset의 {key}클래스 개수: {length_dic[key]}')
        print('---------------------------------------------------------------')
      elif i==1:
        for key in length_dic:
          print( f'validset의 {key}클래스 개수: {length_dic[key]}')
        print('---------------------------------------------------------------')
      else:
        for key in length_dic:
          print( f'validset의 {key}클래스 개수: {length_dic[key]}')
     
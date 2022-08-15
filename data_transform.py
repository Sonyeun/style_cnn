import os
import glob

#사용한 함수
#folder 생성
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except:
        print('Error')


def get_attr(name:str):
    if '.jpg' in name:
        return '.jpg'

    elif '.png' in name:
        return '.png'

    elif '.jpeg' in name:
        return '.jpeg'

    else:
        return None


##데이터 이름 바꾸기##
#새로운 파일 name:  label + artist_name + 걍 index + attr
#aritists 폴더 접근
main_path = 'C:\\Users\\User\\Desktop\\artists\\origin'
artist_lists = os.listdir(main_path)

#artist 폴더 접근
for label, artist in enumerate(artist_lists):
    
    artist_path = main_path + '\\' + artist
    artworks = os.listdir(artist_path)
    

    #files = [artworks.....]
    files = glob.glob(artist_path + '\*')
    for num, f in enumerate(files):
        try:
            attr = get_attr(f.lower())
            new_name = str(label) + artist + '_' + str(num) + attr
            os.rename(f, artist_path + '\\' + new_name)
        except:
            pass



##데이터 스플릿##
## train/ test로 8:2 split
def split(img_list, test_count, train_path, test_path):
  
  test_files=[]
  for i in random.sample( img_list, test_count ):
      if get_attr(i) is not None:
        test_files.append(i)

  # 차집합으로 train/test 리스트 생성하기
  train_files = [x for x in img_list if x not in test_files and get_attr(x) is not None]

  for k in train_files:
    shutil.copy(k, train_path)
  
  for c in test_files:
    shutil.copy(c, test_path)

#aritists 폴더 접근
origin_path = 'C:\\Users\\User\\Desktop\\artists\\origin'
artist_lists = os.listdir(origin_path)

#artist 폴더 접근
for artist in artist_lists:

    #경로 및 폴더 생성
    artist_train_path = 'C:\\Users\\User\\Desktop\\artists\\train'
    artist_test_path = 'C:\\Users\\User\\Desktop\\artists\\test'
    createFolder(artist_train_path)
    createFolder(artist_test_path)
 
    #작품집
    artworks = sorted(glob.glob(origin_path+ '\\' + artist + '\*'))
    
    #train과 test에 얼마나 옮길 지 count, 이거 숫자 정확하지 않다 ~ 중간 과정에서 tiff등은 제외함
    artworks_test_count = round(len(artworks)*0.2)
    print(f'{artist}의 그림 중 test에는 {artworks_test_count} 들어갑니다')
    
    #
    split(artworks, artworks_test_count, artist_train_path, artist_test_path)



##데이터 학습할 수 있는 형태로 변경 및 저장/불러오기##
#근데 이미지 크기가 각기 다르다 보니, batch_size = 1인 형태이다
#1.PIL.Image.open()을 통해 이미지를 불러오고
#2.tensor 형식으로 바꾸고, unsqueeze를 통해 4차원(N,C,H,W)형태로 변경

#데이터 zip 및 불러오기 ~ 각 이미지 크기가 다르다보니 dataloader 불가
#data 경로
train_path = 'C:\\Users\\User\\Desktop\\artists\\train'
test_path = 'C:\\Users\\User\\Desktop\\artists\\test'
train_artworks = glob.glob(train_path + '\*')
test_artworks = glob.glob(test_path + '\*')
pil2tensor = transforms.ToTensor()

#data 라벨링 및 xb,yb 묶기
xb = []
yb = []
i = 1

print(len(train_artworks), '이만큼 해야된다')
with open('C:\\Users\\User\\Desktop\\origin\\train_dl.pkl', 'wb') as fil_e:

    for f in random.sample(train_artworks, len(train_artworks)):
        try:
            img = Image.open(f)
            xb.append(torch.unsqueeze(pil2tensor(img), dim = 0)) 
            yb.append(torch.tensor([int(os.path.basename(f)[0])]))
            i+= 1
    
            if i % 100 == 0:
                print(f'{i}개 만큼 했다')
                train_dl = list(zip(xb,yb))
                pickle.dump(train_dl, fil_e)
                print(f'{i//100}번 째 dataset 만듦')
                xb = []
                yb = []
    
        except:
            print(f)
            pass

    
    print('train에서는 마지막 dataset 저장 간다')
    train_dl = list(zip(xb,yb))  
    pickle.dump(train_dl, fil_e)

#데이터 저장 및 불러오기
print('train_dl 완성완료')
#train_path = 'C:\\Users\\User\\Desktop\\origin\\train_dl.pkl'
#with open('C:\\Users\\User\\Desktop\\origin\\train_dl.pkl', 'rb') as fs:
#    data = pickle.load(fs)

#해당 val_dl 마련 및 저장
xb = []
yb = []
i = 1
num = 0
print('--------')

print(len(test_artworks), '이만큼 해야된다')
with open('C:\\Users\\User\\Desktop\\origin\\val_dl.pkl', 'wb') as fil_e:
    for f in random.sample(test_artworks, len(test_artworks)):
        try:
            img = Image.open(f)
            xb.append(torch.unsqueeze(pil2tensor(img), dim = 0)) 
            yb.append(torch.tensor([int(os.path.basename(f)[0])]))
            i+= 1
    
            if i % 100 == 0:
                print(f'{i}개 만큼 했다')

                val_dl = list(zip(xb,yb))
                pickle.dump(train_dl, fil_e)
                print(f'{i//100}번 째 dataset 만듦')
                xb = []
                yb = []
    
        except:
            print(f)
            pass

    
    val_dl = list(zip(xb,yb))  
    pickle.dump(val_dl, fil_e)

    print('--------------------------------')
    print('--------------------------------')
    print('--------------------------------')
    print('--------------------------------')

    print('all dataset 저장 완료')


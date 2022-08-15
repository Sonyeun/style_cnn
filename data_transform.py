import os
import glob

#����� �Լ�
#folder ����
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


##������ �̸� �ٲٱ�##
#���ο� ���� name:  label + artist_name + �� index + attr
#aritists ���� ����
main_path = 'C:\\Users\\User\\Desktop\\artists\\origin'
artist_lists = os.listdir(main_path)

#artist ���� ����
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



##������ ���ø�##
## train/ test�� 8:2 split
def split(img_list, test_count, train_path, test_path):
  
  test_files=[]
  for i in random.sample( img_list, test_count ):
      if get_attr(i) is not None:
        test_files.append(i)

  # ���������� train/test ����Ʈ �����ϱ�
  train_files = [x for x in img_list if x not in test_files and get_attr(x) is not None]

  for k in train_files:
    shutil.copy(k, train_path)
  
  for c in test_files:
    shutil.copy(c, test_path)

#aritists ���� ����
origin_path = 'C:\\Users\\User\\Desktop\\artists\\origin'
artist_lists = os.listdir(origin_path)

#artist ���� ����
for artist in artist_lists:

    #��� �� ���� ����
    artist_train_path = 'C:\\Users\\User\\Desktop\\artists\\train'
    artist_test_path = 'C:\\Users\\User\\Desktop\\artists\\test'
    createFolder(artist_train_path)
    createFolder(artist_test_path)
 
    #��ǰ��
    artworks = sorted(glob.glob(origin_path+ '\\' + artist + '\*'))
    
    #train�� test�� �󸶳� �ű� �� count, �̰� ���� ��Ȯ���� �ʴ� ~ �߰� �������� tiff���� ������
    artworks_test_count = round(len(artworks)*0.2)
    print(f'{artist}�� �׸� �� test���� {artworks_test_count} ���ϴ�')
    
    #
    split(artworks, artworks_test_count, artist_train_path, artist_test_path)



##������ �н��� �� �ִ� ���·� ���� �� ����/�ҷ�����##
#�ٵ� �̹��� ũ�Ⱑ ���� �ٸ��� ����, batch_size = 1�� �����̴�
#1.PIL.Image.open()�� ���� �̹����� �ҷ�����
#2.tensor �������� �ٲٰ�, unsqueeze�� ���� 4����(N,C,H,W)���·� ����

#������ zip �� �ҷ����� ~ �� �̹��� ũ�Ⱑ �ٸ��ٺ��� dataloader �Ұ�
#data ���
train_path = 'C:\\Users\\User\\Desktop\\artists\\train'
test_path = 'C:\\Users\\User\\Desktop\\artists\\test'
train_artworks = glob.glob(train_path + '\*')
test_artworks = glob.glob(test_path + '\*')
pil2tensor = transforms.ToTensor()

#data �󺧸� �� xb,yb ����
xb = []
yb = []
i = 1

print(len(train_artworks), '�̸�ŭ �ؾߵȴ�')
with open('C:\\Users\\User\\Desktop\\origin\\train_dl.pkl', 'wb') as fil_e:

    for f in random.sample(train_artworks, len(train_artworks)):
        try:
            img = Image.open(f)
            xb.append(torch.unsqueeze(pil2tensor(img), dim = 0)) 
            yb.append(torch.tensor([int(os.path.basename(f)[0])]))
            i+= 1
    
            if i % 100 == 0:
                print(f'{i}�� ��ŭ �ߴ�')
                train_dl = list(zip(xb,yb))
                pickle.dump(train_dl, fil_e)
                print(f'{i//100}�� ° dataset ����')
                xb = []
                yb = []
    
        except:
            print(f)
            pass

    
    print('train������ ������ dataset ���� ����')
    train_dl = list(zip(xb,yb))  
    pickle.dump(train_dl, fil_e)

#������ ���� �� �ҷ�����
print('train_dl �ϼ��Ϸ�')
#train_path = 'C:\\Users\\User\\Desktop\\origin\\train_dl.pkl'
#with open('C:\\Users\\User\\Desktop\\origin\\train_dl.pkl', 'rb') as fs:
#    data = pickle.load(fs)

#�ش� val_dl ���� �� ����
xb = []
yb = []
i = 1
num = 0
print('--------')

print(len(test_artworks), '�̸�ŭ �ؾߵȴ�')
with open('C:\\Users\\User\\Desktop\\origin\\val_dl.pkl', 'wb') as fil_e:
    for f in random.sample(test_artworks, len(test_artworks)):
        try:
            img = Image.open(f)
            xb.append(torch.unsqueeze(pil2tensor(img), dim = 0)) 
            yb.append(torch.tensor([int(os.path.basename(f)[0])]))
            i+= 1
    
            if i % 100 == 0:
                print(f'{i}�� ��ŭ �ߴ�')

                val_dl = list(zip(xb,yb))
                pickle.dump(train_dl, fil_e)
                print(f'{i//100}�� ° dataset ����')
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

    print('all dataset ���� �Ϸ�')


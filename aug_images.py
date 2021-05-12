import os
import argparse
import random
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img,img_to_array,array_to_img,ImageDataGenerator

ap = argparse.ArgumentParser()
ap.add_argument("-op", type=str,required=True,
	help="Output folder for generated images")
ap.add_argument("-map", type=str,required=True,
	help="File for image file to label mapping")
ap.add_argument("-umap", type=str,default=None,required=False,
	help="File for unseen images file to label mapping")
ap.add_argument("-np", type=int,default=20,required=False,
	help="Noise Parameter for Gaussian noise")
ap.add_argument("-aug", type=int,default=1,required=False,
	help="Number of Images for each word/writer")
args = vars(ap.parse_args())

VARIABILITY=args['np']
aug=args['aug']

def add_noise(img):
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

def remove_folder(x):
    x=x.split('/')[::-1]
    return x[0]

def augment_images(train_folder,train_csv_file,train_unseen_csv,aug):
    Y=[]
    df_train=pd.read_csv(train_csv_file)
    k=len(df_train)
    if train_unseen_csv!=None:
        df_random=pd.read_csv(train_unseen_csv)
    else:
        df_random  = pd.DataFrame(columns = ["Image","Word"])
    df_aug = df_train.merge(df_random, how='left', indicator=True)
    df_aug= df_aug[df_aug['_merge'] == 'left_only']
    df_aug = df_aug[['Image', 'Word']]
    df_aug['Image']=train_folder+"/"+df_aug['Image']
    datagen = ImageDataGenerator(
    shear_range=20,
    preprocessing_function=add_noise)
    for i in range(len(df_aug)):
        org=img_to_array(load_img(df_aug['Image'].iloc[i]))
        word=df_aug['Word'].iloc[i]
        cnt=aug
        org = org.reshape((1,) + org.shape)
        for batch in datagen.flow(org, batch_size=1):
            if cnt==0:
                break
            FNi = str(k) + '.png'
            k+=1
            img_path = os.path.join(train_folder, FNi)    
            img=array_to_img(batch[0])
            img.save(img_path)
            Y.append((FNi, word))
            cnt-=1
    df=pd.DataFrame(Y,columns=["Image","Word"])
    df_train=pd.concat([df_train,df],ignore_index=True)
    df_train['Image']=df_train['Image'].apply(remove_folder)
    df_train.set_index('Image', inplace=True)
    df_train.to_csv(train_csv_file)
    
augment_images(args['op'], args['map'], args['umap'], aug)
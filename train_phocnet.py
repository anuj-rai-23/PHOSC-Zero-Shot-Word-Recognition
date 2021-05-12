import os
import argparse
import random
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input,Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU, Activation,BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array,load_img
import matplotlib.pyplot as plt
from phoc_label_generator import gen_phoc_label
from tensorflow.keras.utils import Sequence
from tensorflow_addons.layers import SpatialPyramidPooling2D
os.environ["CUDA_VISIBLE_DEVICES"]="1"
tf.random.set_seed(73)
random.seed(73)
np.random.seed(73) 

ap = argparse.ArgumentParser()
ap.add_argument("-idn", type=str,required=False,
	help="Identifier Name (Prefer Train Set Name)")
ap.add_argument("-batch", type=int,default=10,required=False,
	help="Batch Size")
ap.add_argument("-epoch", type=int,default=20,required=False,
	help="Number of Epochs")
ap.add_argument("-lr", type=float,default=1e-4,required=False,
	help="Learning rate for optimizer")
ap.add_argument("-mp", type=str,required=True,
	help="CSV file for Train Image to Class Label map")
ap.add_argument("-vi", type=str,required=True,
	help="Folder for Validation Images")
ap.add_argument("-vmap", type=str,required=True,
	help="CSV file for Validation Image to Class Label map")
ap.add_argument("-umap", type=str,default=None,required=False,
	help="CSV file for Unseen images to Class Label map")
ap.add_argument("-tr", type=str,required=True,
	help="Folder having Train Images")

args = vars(ap.parse_args())

#print(args)

MODEL=args['idn']
BATCH_SIZE=args['batch']
EPOCHS=args['epoch']
LR=args['lr']
train_csv_file=args['mp']
valid_csv_file=args['vmap']
train_unseen_csv_file=args['umap']
train_folder=args['tr']
valid_folder=args['vi']
model_name="new_"+MODEL+"_"+str(BATCH_SIZE)+"_"

class DataSequence(Sequence):
    def __init__(self, df, batch_size):
        self.df = df # your pandas dataframe
        self.bsz = batch_size # batch size
        
        # Take labels and a list of image locations in memory
        self.labels = np.asarray(self.df['Label'].tolist()).astype(np.float32)
        self.im_list = self.df['Image'].tolist()

    def __len__(self):
        # compute number of batches to yield
        return int(math.ceil(len(self.df) / float(self.bsz)))

    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        self.indexes = range(len(self.im_list))
        self.indexes = random.sample(self.indexes, k=len(self.indexes))
        
    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx * self.bsz: (idx + 1) * self.bsz])

    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        return np.array([img_to_array(load_img(im)) for im in self.im_list[idx * self.bsz: (1 + idx) * self.bsz]])

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y

'''
MODEL=None
BATCH_SIZE=64
EPOCHS=100
train_csv_file='train_report.csv'
valid_csv_file='valid_report.csv'
train_folder='train_img'
valid_folder='valid_img'
'''

def build_phocnet():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu',input_shape=(None,None,3)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(SpatialPyramidPooling2D([1,2,4]))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(4096, activation='linear'))
    model.add(Dense(604, activation='sigmoid'))
    
    #optimizer = tf.keras.optimizers.SGD(lr=1e-4, momentum=.9, decay=5e-5)
    loss = tf.keras.losses.binary_crossentropy
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-07,decay=5e-5)
    model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.CosineSimilarity(axis=1)])
    model.summary()
    return model

def getlabel(x):
    return all_labels[x]
    


df_train=pd.read_csv(train_csv_file)
df_valid=pd.read_csv(valid_csv_file)
if train_unseen_csv_file!=None:
    df_unseen=pd.read_csv(train_unseen_csv_file)
    df_train = df_train.merge(df_unseen, how='left', indicator=True)
    df_train= df_train[df_train['_merge'] == 'left_only']
    df_train = df_train[['Image', 'Word']]
if train_folder==valid_folder:
    df_train = df_train.merge(df_valid, how='left', indicator=True)
    df_train= df_train[df_train['_merge'] == 'left_only']
    df_train = df_train[['Image', 'Word']]

print("Train_Images=",len(df_train),"Valid_Images=",len(df_valid))

train_word_label=gen_phoc_label(list(set(df_train['Word'])))
valid_word_label=gen_phoc_label(list(set(df_valid['Word'])))
all_labels={**train_word_label,**valid_word_label}
df_train['Image']=train_folder+"/"+df_train['Image']
df_valid['Image']=valid_folder+"/"+df_valid['Image']
df_train['Label']=df_train['Word'].apply(getlabel)
df_valid['Label']=df_valid['Word'].apply(getlabel)
model=build_phocnet()

train_sequence = DataSequence(df_train, BATCH_SIZE)
valid_sequence = DataSequence(df_valid, BATCH_SIZE)     
STEPS=len(df_train)//BATCH_SIZE
EPOCHS=70000//STEPS+10000//STEPS+1

def learning_rate_scheduler(epoch, lr):
    #decay_rate = 1.1
    #decay_step = 2
    if epoch > 70000//STEPS:
        return 1e-5
    return lr

print("Model Built")

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=5, verbose=2,mode='auto', baseline=None, restore_best_weights=False)
lrs=tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler, verbose=0)
rlp=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=3, verbose=1,mode='auto', min_delta=1e-7, cooldown=3, min_lr=1e-7)
callbacks_list = [early_stop,rlp]
history=model.fit(train_sequence, epochs=EPOCHS, validation_data=valid_sequence,shuffle=True,callbacks=callbacks_list)
model.save(model_name+".h5")

if not os.path.exists("Train_History"):
    os.makedirs("Train_History") 

hist_df = pd.DataFrame(history.history) 
hist_csv_file = 'Train_History/history_'+model_name+'.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

acc = history.history['cosine_similarity']
val_acc = history.history['val_cosine_similarity']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc,label='Training Similarity')
plt.plot(epochs, val_acc,label='Validation Similarity')
plt.title(model_name+'_Cosine Similarity')
plt.legend()
plt.savefig('Train_History/'+model_name+'_Pretrain_CS.png')
plt.show()
plt.plot(epochs, loss,label='Training Loss')
plt.plot(epochs, val_loss,label='Validation Loss')
plt.title(model_name+' MSE Loss')
plt.legend()
plt.savefig('Train_History/'+model_name+'_Pretrain_Loss.png')
plt.show()

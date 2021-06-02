# Library imports 

import os
import argparse
import random
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input,Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU, Activation,BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import  ResNet50
from tensorflow.keras.preprocessing.image import img_to_array,ImageDataGenerator,load_img
import matplotlib.pyplot as plt
from phos_label_generator import gen_label
from phoc_label_generator import gen_phoc_label
from tensorflow.keras.utils import Sequence
from tensorflow_addons.layers import SpatialPyramidPooling2D

# Uncomment the following line and set appropriate GPU if you want to set up/assign a GPU device to run this code
# os.environ["CUDA_VISIBLE_DEVICES"]="1"	

# Setting random seeds
tf.random.set_seed(73)
random.seed(73)
np.random.seed(73) 


ap = argparse.ArgumentParser()
ap.add_argument("-idn", type=str,required=False,
	help="Identifier Name (Prefer Train Set Name)")
ap.add_argument("-batch", type=int,default=64,required=False,
	help="Batch Size")
ap.add_argument("-epoch", type=int,default=40,required=False,
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


# DataSequence class to pass data(images/vector) in batches

class DataSequence(Sequence):
    def __init__(self, df, batch_size):
        self.df = df # your pandas dataframe
        self.bsz = batch_size # batch size
        
        # Take labels and a list of image locations in memory
        self.labels=[]
        for i in range(len(self.df)):
            self.labels.append({"phosnet":np.asarray(self.df['PhosLabel'].iloc[i]).astype(np.float32),"phocnet":np.asarray(self.df['PhocLabel'].iloc[i]).astype(np.float32)})
        print(len(self.labels))
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
        l1=[]
        l2=[]
        for x in batch_y:
            l1.append(x['phosnet'])
            l2.append(x['phocnet'])
        #return batch_x, batch_y
        return batch_x,{'phosnet':np.asarray(l1),'phocnet':np.asarray(l2)}

# Function to build and return SPP-Pho(SC)Net model 

def build_model():
    inp = Input(shape=(None,None,3))
    model=Conv2D(64, (3, 3), padding='same',activation='relu')(inp)
    model=Conv2D(64, (3, 3), padding='same', activation='relu')(model)
    model=(MaxPooling2D(pool_size=(2, 2), strides=2))(model)
    model=(Conv2D(128, (3, 3), padding='same', activation='relu'))(model)
    model=(Conv2D(128, (3, 3), padding='same', activation='relu'))(model)
    model=(MaxPooling2D(pool_size=(2, 2), strides=2))(model)
    model=(Conv2D(256, (3, 3), padding='same', activation='relu'))(model)
    model=(Conv2D(256, (3, 3), padding='same', activation='relu'))(model)
    model=(Conv2D(256, (3, 3), padding='same', activation='relu'))(model)
    model=(Conv2D(256, (3, 3), padding='same', activation='relu'))(model)
    model=(Conv2D(256, (3, 3), padding='same', activation='relu'))(model)
    model=(Conv2D(256, (3, 3), padding='same', activation='relu'))(model)
    model=(Conv2D(512, (3, 3), padding='same', activation='relu'))(model)
    model=(Conv2D(512, (3, 3), padding='same', activation='relu'))(model)
    model=(Conv2D(512, (3, 3), padding='same', activation='relu'))(model)
    model=(SpatialPyramidPooling2D([1,2,4]))(model)
    model=(Flatten())(model)
    
	phosnet_op=Dense(4096, activation='relu')(model)
    phosnet_op=Dropout(0.5)(phosnet_op)
    phosnet_op=Dense(4096, activation='relu')(phosnet_op)
    phosnet_op=Dropout(0.5)(phosnet_op)
    phosnet_op=Dense(165, activation='relu',name="phosnet")(phosnet_op)

    phocnet=Dense(4096, activation='relu')(model)
    phocnet=Dropout(0.5)(phocnet)
    phocnet=Dense(4096, activation='relu')(phocnet)
    phocnet=Dropout(0.5)(phocnet)
    phocnet=Dense(604, activation='sigmoid',name="phocnet")(phocnet)

    model = Model(inputs=inp, outputs=[phosnet_op,phocnet])
    losses = {
    "phosnet": tf.keras.losses.MSE,
    "phocnet": 'binary_crossentropy',
    }
    lossWeights = {"phosnet": 1.5, "phocnet": 4.5}
    # initialize the optimizer and compile the model

    opt = tf.keras.optimizers.Adam(lr=LR,decay=5e-5)
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,metrics=[tf.keras.metrics.CosineSimilarity(axis=1)])
	#model.summary()
    return model

def getphoclabel(x):
    return all_phoc_labels[x]

def getphoslabel(x):
    return all_phos_labels[x]


# Build train set dataframe after removing validation set and seen class test set images from train 

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

# Generating dictionaries of words mapped to PHOS & PHOC vectors
train_word_phos_label=gen_label(list(set(df_train['Word'])))
valid_word_phos_label=gen_label(list(set(df_valid['Word'])))
all_phos_labels={**train_word_phos_label,**valid_word_phos_label}
train_word_phoc_label=gen_phoc_label(list(set(df_train['Word'])))
valid_word_phoc_label=gen_phoc_label(list(set(df_valid['Word'])))
all_phoc_labels={**train_word_phoc_label,**valid_word_phoc_label}

# Adding folder names to file names
df_train['Image']=train_folder+"/"+df_train['Image']
df_valid['Image']=valid_folder+"/"+df_valid['Image']
df_train['PhosLabel']=df_train['Word'].apply(getphoslabel)
df_valid['PhosLabel']=df_valid['Word'].apply(getphoslabel)
df_train['PhocLabel']=df_train['Word'].apply(getphoclabel)
df_valid['PhocLabel']=df_valid['Word'].apply(getphoclabel)

# Build model
model=build_model()
print("Model Built")

# Sequence for passing data(images, PHOS labels) to model
train_sequence = DataSequence(df_train, BATCH_SIZE)
valid_sequence = DataSequence(df_valid, BATCH_SIZE) 

# Early stopping and ReduceLROnPlateau callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=2,mode='auto', baseline=None, restore_best_weights=False)
rlp=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_phocnet_loss', factor=0.25, patience=5, verbose=1,mode='auto', min_delta=0.0001, cooldown=2, min_lr=1e-7)
callbacks_list = [early_stop,rlp]

# Training the model 
history=model.fit(train_sequence, epochs=EPOCHS, validation_data=valid_sequence,shuffle=True,callbacks=callbacks_list)

# Save the model after training completes
model.save(model_name+".h5")

# Create directory to store training history
if not os.path.exists("Train_History"):
    os.makedirs("Train_History") 

# Store train history as CSV file
hist_df = pd.DataFrame(history.history) 
hist_csv_file = 'Train_History/history_'+model_name+'.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# Plot train and validation accuracy(avg cosine similarity)
acc = history.history['phocnet_cosine_similarity']
val_acc = history.history['val_phocnet_cosine_similarity']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc,label='Training Similarity')
plt.plot(epochs, val_acc,label='Validation Similarity')
plt.title(model_name+'_Cosine Similarity')
plt.legend()
plt.savefig('Train_History/'+model_name+'_Pretrain_CS.png')
plt.show()

# Plot train and validation loss
plt.plot(epochs, loss,label='Training Loss')
plt.plot(epochs, val_loss,label='Validation Loss')
plt.title(model_name+' MSE Loss')
plt.legend()
plt.savefig('Train_History/'+model_name+'_Pretrain_Loss.png')
plt.show()


# Library imports 

import os
import argparse
import itertools
import numpy as np
from numpy import linalg as LA
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU, Activation,BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import  ResNet50
from tensorflow.keras.preprocessing.image import img_to_array,load_img
import matplotlib.pyplot as plt
from phos_label_generator import gen_label
from phoc_label_generator import gen_phoc_label
from tensorflow_addons.layers import SpatialPyramidPooling2D

# Uncomment the following line and set appropriate GPU if you want to set up/assign a GPU device to run this code
# os.environ["CUDA_VISIBLE_DEVICES"]="1"	

# Argument parser variables

ap = argparse.ArgumentParser()
ap.add_argument("-model", type=str,required=False,
	help="Pretrained Model")
ap.add_argument("-test", type=str,required=True,
	help="Folder having unseen words")
ap.add_argument("-mp", type=str,required=True,
	help="CSV file for Test Images to Class Label map")
ap.add_argument("-stf", type=str,default=None,required=False,
	help="Folder for seen Words (Gen. ZSl Setting)")
ap.add_argument("-smap", type=str,default=None,required=False,
	help="CSV file for Seen Images to Class Label map (Gen. ZSL Setting)")
ap.add_argument("-train", type=str,default=None,required=False,
	help="CSV file for Train Images to Class Label map (Gen. ZSL Setting)")
ap.add_argument("-idn", type=str,required=False, default='',
	help="Identifier for saving image files")
args = vars(ap.parse_args())

# Input: Two vectors x and y
# Output: Similarity index = Cosine simialirity * 1000

def similarity(x,y):
    return 1000*np.dot(x,y)/(LA.norm(x)*LA.norm(y))

# Function to build and return Pho(SC) model 

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
	
	# PHOS component
    phosnet_op=Dense(4096, activation='relu')(model)
    phosnet_op=Dropout(0.5)(phosnet_op)
    phosnet_op=Dense(4096, activation='relu')(phosnet_op)
    phosnet_op=Dropout(0.5)(phosnet_op)
    phosnet_op=Dense(165, activation='relu',name="phosnet")(phosnet_op)

	# PHOC component
    phocnet=Dense(4096, activation='relu')(model)
    phocnet=Dropout(0.5)(phocnet)
    phocnet=Dense(4096, activation='relu')(phocnet)
    phocnet=Dropout(0.5)(phocnet)
    phocnet=Dense(604, activation='sigmoid',name="phocnet")(phocnet)
    model = Model(inputs=inp, outputs=[phosnet_op,phocnet])
    return model

# Input: Confusion matrix, true and predicted class names, plot title, color map and normalization parameter(bool)
# Output: Plots and saves confusion matrix 

def plot_confusion_matrix(cm,target_names_true,target_names_pred,title='Confusion matrix',cmap=None,normalize=True):
    #accuracy = np.trace(cm) / float(np.sum(cm))

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm * 100
    cm[np.isnan(cm)] = 0
    plt.figure(figsize=(12, 9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks_true = np.arange(len(target_names_true))
    tick_marks_pred = np.arange(len(target_names_pred))
    plt.xticks(tick_marks_pred, target_names_pred)
    plt.yticks(tick_marks_true, target_names_true)


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True Word Class Label Length')
    plt.savefig("Test_Plots/"+title+".png")
    plt.xlabel('Predicted Word Class Label Length')
    plt.show()

# Input: Model, dataframes for test set samples, dictionary for test set words and label(PHOC vector)
# Output: Similarity index = Cosine simialirity * 1000

def accuracy_test(model,df_test,test_word_label,name):
    cnt=0
    no_of_images=len(df_test)
    acc_by_len=dict()
    word_count_by_len=dict()
    for k in df_test['Word'].tolist():
        acc_by_len[len(k)]=0
        word_count_by_len[len(k)]=0
    lengths_true=sorted(acc_by_len.keys())
    lengths_pred=list(set(len(x) for x in test_word_label))
    l=len(lengths_true)
    m=len(lengths_pred)
    idx_true=dict()
    idx_pred=dict()
    for k in range(l):
        idx_true[lengths_true[k]]=k
    for k in range(m):
        idx_pred[lengths_pred[k]]=k
    conf_matrix=np.zeros(shape=(l,m))
    Predictions=[]
    for i in range(len(df_test)):
        #print(i)
        x=img_to_array(load_img(df_test['Image'].iloc[i]))
        word=df_test['Word'].iloc[i]
        word_count_by_len[len(word)]+=1
        x = np.expand_dims(x, axis=0)
        y_pred=model.predict(x)
        y_pred=np.squeeze(np.concatenate((y_pred[0],y_pred[1]),axis=1))
        #print(y_pred)
        mx=0
        for k in test_word_label:
            temp=similarity(y_pred,test_word_label[k])
            if temp>mx:
                mx=temp
                op=k
        #print(word,op,mx)
        conf_matrix[idx_true[len(word)]][idx_pred[len(op)]]+=1
        Predictions.append((df_test['Image'].iloc[i],word,op))
        if op==word:
            cnt+=1
            acc_by_len[len(word)]+=1
    for k in acc_by_len:
        if acc_by_len[k]!=0:
            acc_by_len[k]=acc_by_len[k]/word_count_by_len[k] * 100
    df=pd.DataFrame(Predictions,columns=["Image","True Label","Predicted Label"])
    df.set_index('Image', inplace=True)
    df.to_csv("Test_Results/"+name+".csv")
    print("Correct predictions:",cnt,"   Accuracy=",cnt/no_of_images)
    plt.figure(figsize=(10,6))
    plt.bar(*zip(*acc_by_len.items()))
    plt.title('Acc:'+str(cnt)+'/'+str(no_of_images)+'  Correct predictions lengthwise')
    plt.xticks(lengths_true)
    plt.xlabel('Word Length')
    plt.ylabel('Percentage of correct predictions')
    plt.savefig("Test_Plots/"+name+"_ZSL_acc.png")
    plt.show()
    plot_confusion_matrix(conf_matrix,lengths_true,lengths_pred,title=name+"_confmat")
    return cnt/no_of_images

# Input: A word(string) 
# Output: Pho(SC) vector after concatenating PHOS and PHOC 

def get_comb_label(x):
    phos_labels=gen_label(x)
    phoc_labels=gen_phoc_label(x)
    test_labels=dict()
    for x in phos_labels:
        test_labels[x]=np.concatenate((phos_labels[x],phoc_labels[x]),axis=0)
    return test_labels

# Input: model, folder names for samples, CSV files having sample to label mapping(Train and test set), and name(identifier in plot names)
# Output: Prediction accuracy (Also calls functions for plotting)

def zsl_test(model,test_folder,test_csv_file,seen_word_folder,seen_word_map,train_csv_file,name):
    df_test=pd.read_csv(test_csv_file)
    test_word_label=get_comb_label(list(set(df_test['Word'])))
    df_test['Image']=test_folder+"/"+df_test['Image']
    acc_unseen=accuracy_test(model, df_test, test_word_label, name+"_conv")
    print("Conventional ZSL Accuracy = ", acc_unseen)

    if seen_word_folder!=None and train_csv_file!=None:
        df_train=pd.read_csv(seen_word_map)
        df_lex=pd.read_csv(train_csv_file)
        train_word_label=get_comb_label(list(set(df_lex['Word'])))
        df_train['Image']=seen_word_folder+"/"+df_train['Image']
        test_word_label={**test_word_label,**train_word_label}
        acc_unseen=accuracy_test(model, df_test, test_word_label, name+"_gen_unseen")
        acc_seen=accuracy_test(model, df_train, test_word_label, name+"_gen_seen")
        print("Accuracy with Unseen Words = ", acc_unseen)
        print("Accuracy with Seen Words = ", acc_seen)
        gen_zsl_acc=2*acc_unseen*acc_seen/(acc_unseen+acc_seen)
        print("Generalized ZSL Accuracy = ",gen_zsl_acc)



MODEL=args['model']
test_folder=args['test']
test_map=args['mp']
seen_word_map=args['smap']
seen_words_folder=args['stf']
train_map=args['train']
name=MODEL+"_"+args['idn']

# Create directories for storing test results and plots

if not os.path.exists("Test_Plots"):
    os.makedirs("Test_Plots")
if not os.path.exists("Test_Results"):
    os.makedirs("Test_Results")

# Build model and load weights from filename and print model name(if successfully loaded)
model=build_model()
model.load_weights(MODEL+".h5")
#model=tf.keras.models.load_model(MODEL+".h5")
print(MODEL)

# Function called for test set prediction and result plotting
zsl_test(model,test_folder,test_map,seen_words_folder,seen_word_map,train_map,name)


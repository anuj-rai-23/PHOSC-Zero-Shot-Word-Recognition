import os
import argparse
import itertools
import numpy as np
from numpy import linalg as LA
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array,load_img
import matplotlib.pyplot as plt
from tensorflow_addons.layers import SpatialPyramidPooling2D
from phos_label_generator import gen_label
os.environ["CUDA_VISIBLE_DEVICES"]="2"
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

def similarity(x,y):
    return 1000*np.dot(x,y)/(LA.norm(x)*LA.norm(y))


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
    plt.savefig("X_Test_Plots/"+title+".png")
    plt.xlabel('Predicted Word Class Label Length')
    plt.show()

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
        y_pred=np.squeeze(model.predict(x))
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
    df.to_csv("X_Test_Results/"+name+".csv")
    print("Correct predictions:",cnt,"   Accuracy=",cnt/no_of_images)
    plt.figure(figsize=(10,6))
    plt.bar(*zip(*acc_by_len.items()))
    plt.title('Acc:'+str(cnt)+'/'+str(no_of_images)+'  Correct predictions lengthwise')
    plt.xticks(lengths_true)
    plt.xlabel('Word Length')
    plt.ylabel('Percentage of correct predictions')
    plt.savefig("X_Test_Plots/"+name+"_ZSL_acc.png")
    plt.show()
    plot_confusion_matrix(conf_matrix,lengths_true,lengths_pred,title=name+"_confmat")
    return cnt/no_of_images


def zsl_test(model,test_folder,test_csv_file,seen_word_folder,seen_word_map,train_csv_file,name):
    df_test=pd.read_csv(test_csv_file)
    test_word_label=gen_label(list(set(df_test['Word'])))
    df_test['Image']=test_folder+"/"+df_test['Image']
    acc_unseen=accuracy_test(model, df_test, test_word_label, name+"_conv")
    print("Conventional ZSL Accuracy = ", acc_unseen)

    if seen_word_folder!=None and train_csv_file!=None:
        df_train=pd.read_csv(seen_word_map)
        df_lex=pd.read_csv(train_csv_file)
        train_word_label=gen_label(list(set(df_lex['Word'])))
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

if not os.path.exists("X_Test_Plots"):
    os.makedirs("X_Test_Plots")
if not os.path.exists("X_Test_Results"):
    os.makedirs("X_Test_Results")


model=tf.keras.models.load_model(MODEL+".h5")
print(MODEL)
zsl_test(model,test_folder,test_map,seen_words_folder,seen_word_map,train_map,name)



#generalized_zsl_test(model,test_folder,train_csv,test_csv,name)

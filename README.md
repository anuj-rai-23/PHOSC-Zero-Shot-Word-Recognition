# PHOSC-Zero-Shot-Word-Recognition

![SPP-Pho(SC)Net Image](Images/Pho(SC)net_Architecture.png)

**PHOSNet** and **Pho(SC)Net** are convolution neural networks for characterizing document word images using proposed *PHOS* and *Pho(SC)* embedding, which can be used for word recognition in Zero-shot setting. This code was used for generating results for the work [Pho(SC)Net: An Approach Towards Zero-shot Word Image Recognition in Historical Documents](https://link.springer.com/chapter/10.1007/978-3-030-86549-8_2).

Implemented using Tensorflow 2.x framework.

## Requirements
- Python 3.8
- Tensorflow 2.x 
- Numpy
- Pandas
- Matplotlib

## Files 
| File Name | Significance |
| ------ | ------ |
| Alphabet.csv | Stores count of primary shapes for alphabets |
| aug_images.py | Augmentation of the images in folder (also saves them in same folder)|
| phoc_label_generator.py | Generating PHOC vectors |
| phos_label_generator.py | Generating PHOS vectors |
| train_\<model\>.py | Training the model |
| test_\<model\>.py | Testing the performance of saved model |
> Saved model names follow a naming convention, i.e. *new_(model_identifier)_(batch_size)\_.h5*

#### Additional Directories generated after training/testing
| Directory | Significance |
| ------ | ------ |
| Train_History | Saves the training data (losses/similarities) for each epoch |
| Test_Results | Saves true and predicted labels for each sample in test set|
| Test_Plots | Saves plots generated by testing module|

>Plots generated during testing are named as follows: *(model_name)_(test_identifier)\_(test_setting).png*

>Test result files are named as follows: *(model_name)_(test_identifier)\_(test_setting).csv*

## Dataset
We are providing the dataset used ( along with the augmented images) 
* [Standard Split](https://mega.nz/file/i6JFGSwS#vrBJpgZu4yHZELs5fTjH5BvLXX81UUnckvtjPqWl9aw)
* [ZSL Split](https://mega.nz/file/6mBCjSAL#Hz2xJtoWUJXuC9bf4E-4jP4HrhgAks6jT_2RHnOnmgM)

## Code Execution
Consider organizing your dataset in the following directory structure:

📦 Dataset</br>
 ┣ 📦  Train </br>
 ┣ 📦  Validation</br>
 ┣ 📦  Test</br> 
 ┣ 📜  Alphabet.csv </br> 
 ┣ 📜  Test_seen.csv </br>
 ┣ 📜  Test_unseen.csv </br>
 ┣ 📜  Train.csv </br>
 ┣ 📜  Train_unseen.csv </br>
 ┣ 📜  Valid_seen.csv </br>
 ┗ 📜  Valid_unseen.csv </br>

##### Execute the following :
* For data augmentation </br>
`python aug_images.py -op Dataset/Validation -map Train_unseen.csv -np 40 -aug 1` </br>
 Please perform this after going through code once. It will add n copies of already existing images to the same folder.</br>
 File Name assignment is done using the number, e.g., if there are 20 images already in the folder, new augmented images will be as 21.png, 22.png .........

* For training a model </br>
`python train_phoscnet.py -idn MySimpleDataset -batch 128 -lr 0.0005 -epoch 200 -mp Dataset/Train.csv -vmap Dataset/Valid_seen.csv   -tr Dataset/Train -vi Dataset/Validation` </br>

* For testing the saved model </br>
`python test_phoscnet.py -model GW_split1_64.h5 -idn GenConv -mp Test_seen.csv -stf Dataset/Train -smap Dataset/Train_unseen.csv   -train Dataset/Train`
## Results
Following image shows prediction results on images from IAM-handwriting dataset.
![SPP-Pho(SC)Net Image](Images/PredictionExamples.png)

## Citation
If you find the paper or the source code useful to your projects, please cite the following bibtex: 
<pre>
@InProceedings{10.1007/978-3-030-86549-8_2,
author="Rai, Anuj and Krishnan, Narayanan C. and Chanda, Sukalpa",
title="Pho(SC)Net: An Approach Towards Zero-Shot Word Image Recognition in Historical Documents",
booktitle="Document Analysis and Recognition -- ICDAR 2021",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="19--33",
isbn="978-3-030-86549-8"
}
</pre>

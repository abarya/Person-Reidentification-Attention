#!/usr/bin/python
# encoding=utf8
import numpy as np
import cv2
import os
import shutil
#src="/home/arya/Documents/Attentional_person_reid/datasets/finetune_dataset/cuhk01_test100/train/000100"
#dst="/home/arya/Documents/Attentional_person_reid/datasets/finetune_dataset/cuhk01_test100/train_new/001"
#shutil.copytree(src, dst)

#this part divides the images as per their identity in the dataset
#in the labelled_dir, it stores images of a particular identity in the corresponding sub-directory 
'''img_dir="datasets/cuhk01_test100/campus/"
image_list=[x for x in os.listdir("datasets/cuhk01_test100/campus")]
for i in range(len(image_list)):
	final_dir="datasets/cuhk01_test100/labelled_dir/"+image_list[i][:4]
	if not os.path.exists(final_dir):
	    os.makedirs(final_dir)	
	img=cv2.imread("datasets/cuhk01_test100/campus/"+image_list[i])#img_dir+image_list[0])
	if img is not None :
		cv2.imwrite(final_dir+"/"+image_list[i],img) 
'''

#this part resizes the image to suit the CNN model

'''

directory="datasets/cuhk01_test100/labelled_dir/"

sub_directories=[x[0] for x in os.walk(directory)][1:]

for i in range(len(sub_directories)):
	img_dir=sub_directories[i]
	image_list=[x for x in os.listdir(img_dir)]
	for j in range(len(image_list)):
		img=cv2.imread(sub_directories[i]+"/"+image_list[j])#img_dir+image_list[0])
		img=cv2.resize(img,(227,227))
		cv2.imwrite(sub_directories[i]+"/"+image_list[j],img)

'''

#creating a txt file that will contain the paths to each image in the train and test data	
'''f = open('datasets/cuhk01_test100.txt','w') 
directory="datasets/cuhk01_test100/labelled_dir/"
sub_directories=[x[0] for x in os.walk(directory)][1:]
for i in range(len(sub_directories)):
	img_dir=sub_directories[i]
	
	image_list=[x for x in os.listdir(img_dir)]
	for j in range(len(image_list)):
		f.write(sub_directories[i]+"/"+image_list[j]+" "+img_dir[-4:]+"\n")
		#print sub_directories[i]+"/"+image_list[j]

'''
'''directory="/home/arya/Documents/Attentional_person_reid/datasets/finetune_dataset/cuhk01_test100/train"
final_dir="/home/arya/Documents/Attentional_person_reid/datasets/finetune_dataset/cuhk01_test100/train_new/"
sub_directories=[x[0] for x in os.walk(directory)][1:]
for i in range(len(sub_directories)):
	x="%04d" % (i+1)
	src=sub_directories[i]
	dst=final_dir+str(x)
	shutil.copytree(src, dst)
'''

directory="/home/arya/Documents/Attentional_person_reid/datasets/finetune_dataset/cuhk01_test100/train_new"
sub_directories=[x[0] for x in os.walk(directory)][1:]
print directory[:-9]
for i in range(len(sub_directories)):
	img_dir=sub_directories[i]
	image_list=[x for x in os.listdir(img_dir)]
	x=np.random.permutation(len(image_list))
	if not os.path.exists(directory[:-4]+"/"+sub_directories[i][-4:]):
	    os.makedirs(directory[:-4]+"/"+sub_directories[i][-4:])
	for j in range(24):
		img=cv2.imread(sub_directories[i]+"/"+image_list[j])#img_dir+image_list[0])
		cv2.imwrite(directory[:-4]+"/"+sub_directories[i][-4:]+"/"+image_list[j],img)
	if not os.path.exists(directory[:-9]+"/"+"test"+"/"+sub_directories[i][-4:]):
	    os.makedirs(directory[:-9]+"/"+"test"+"/"+sub_directories[i][-4:])
	for j in range(24,28):
		img=cv2.imread(sub_directories[i]+"/"+image_list[j])#img_dir+image_list[0])
		cv2.imwrite(directory[:-9]+"/"+"test"+"/"+sub_directories[i][-4:]+"/"+image_list[j],img)
			
	




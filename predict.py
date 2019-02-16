#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:37:36 2019 by @nirajmodi 
Last Modified on Feb 16 17:59:36 2019 @aakash30jan
"""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import cv2
from subprocess import call
import os

ap = argparse.ArgumentParser()  
ap.add_argument("-i", "--image", required = True, help = "Path to input image")
args = vars(ap.parse_args())

imagePath=args["image"] #path - remote or local

if imagePath.find('http') > -1 or imagePath.find('www') > -1:  #if remote path
    if os.path.isdir('./wgetDir') != True:
         print("Creating saveDir='./wgetDir'")
         os.mkdir('./wgetDir')        
    print("w-getting ",imagePath)
    location=imagePath
    saveName=location[::-1][:location[::-1].find('/')][::-1]
    os.chdir('./wgetDir')
    cmd='wget '+location
    call(cmd,shell=True)
    os.chdir('../')
    imagePath='./wgetDir/'+saveName
else:
    saveName= imagePath 
    
image = cv2.imread(imagePath) #load from path    
#image = cv2.imread(args["image"]) #load from path    
image = cv2.resize(image, (100, 100))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis = 0)

model = load_model('VadaPav_modified.model')

(notVadaPav, VadaPav) = model.predict(image)[0]
ansNO,ansYES=notVadaPav, VadaPav

label = "a Vada-Pav" if VadaPav > notVadaPav else "NOT a Vada-Pav"
proba = VadaPav if VadaPav > notVadaPav else notVadaPav
proba *= 100
print("#"*8+" Prediction "+"#"*8)
print("\n The given image is/has {} with {:.2f}% probability". format(label, proba))
print("#"*28)

import datetime
stamp=str(datetime.datetime.now()).split('.')[0] #local-time of system
if os.path.isfile('./wgetDir/logFile.txt') == False:
         print("Creating logFile='./wgetDir/logFile.txt")
         call("touch ./wgetDir/logFile.txt",shell=True)   
         data=open('./wgetDir/logFile.txt','a+')
         data.write("#"+stamp+" \t "+"filename \t probabilityYES \t probabilityNO \n")
         data.close()
if os.path.isfile('./wgetDir/logFile.txt') == True:
    data=open('./wgetDir/logFile.txt','a+')
    line=stamp+" \t "+ saveName+" \t "+str(ansYES)+" \t "+str(ansNO)+" \n"
    data.write(line)
    data.close()         




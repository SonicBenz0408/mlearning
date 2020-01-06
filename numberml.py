import cv2
import numpy as np
from sklearn.datasets import load_digits
digits = load_digits()
digits.keys()
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(digits.data,digits.target,random_state=0)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain,ytrain)

img_paths=[[0,'0.png'],[1,'1.png'],[2,'2.png'],[3,'3.png'],[4,'4.png'],
           ['white0','white0.png'],['white1','white1.png'],['white2','white2.png'],
           ['white3','white3.png'],['white4','white4.png'],['white5','white5.png'],
           ['white6','white6.png'],['white7','white7.png'],['white8','white8.png'],
           ['white9','white9.png']]
for image in img_paths:
    img_path = image[1]
    img = cv2.imread(img_path, 0)
    img_reverted= cv2.bitwise_not(img)
    new_img = img_reverted / 255.0 *16
    result=new_img.flatten()
    pred=model.predict([result])
    print('it predict %s as %d'%(image[0],pred))





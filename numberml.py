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

img_paths=['0.png','1.png','2.png','3.png']
for image in img_paths:
    img_path = image
    img = cv2.imread(img_path, 0)
    img_reverted= cv2.bitwise_not(img)
    new_img = img_reverted / 255.0 *16
    result=new_img.flatten()
    pred=model.predict([result])
    print(pred)





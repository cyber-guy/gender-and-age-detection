#Importing necessary libraries. 
import argparse
import cv2
from cv2 import CascadeClassifier
from cv2 import CascadeClassifier_convert
import numpy
import pandas 
import matplotlib.pyplot as plt 
from skimage.io import imread, imshow 
from skimage.transform import resize


parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()


ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"


#Loading the different trained models for gender and age determination 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
age_model=cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel') 
gender_model =cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')

#Values used in the pretrained model

mean_values = (78.4263377603, 87.7689143744, 114.895847746) 
age_values= [' (0, 2)', '(4, 6)', '(8,12)', '(15, 17)', '(21, 25)', '(60,80)'] 
gender_values = ['Male', 'Female']

#Reading the image
src = cv2.imread ('grish.jpeg') 
newimage = cv2.cvtColor(src, cv2.COLOR_BGR2RGB )


# #Cropping the face from the image using face algorithm.
minisize = (newimage.shape[1] , newimage.shape[0] )
miniframe =  cv2.resize (newimage, minisize)

faces = face_cascade.detectMultiScale(image = newimage, scaleFactor=1.25, minNeighbors=3, minSize=(40,40)) 
# facedata = "haarcascade_frontalface_default.xml"
# cascade = cv2.CascadeClassifier (facedata)
for (x, y, w, h ) in faces:
    cv2.rectangle(newimage, (x, y) , (x+w, y+h), (255, 255, 0), 2)
    new_img = newimage [y:y+h, x:x+w].copy()
    img_blob=cv2.dnn.blobFromImage(new_img, 1, (227, 227) , mean_values, swapRB=False)

#Predicting gender
plt.figure(1, figsize=(20,20))
gender_model.setInput (img_blob)
gender_predicted = gender_model.forward()
gender = gender_values[gender_predicted[0].argmax()]
print ("Gender: "+ gender)
print ("Predicted values of gender are: ", gender_predicted)

#Predicting age
age_model.setInput(img_blob)
age_predicted = age_model.forward()
age = age_values [age_predicted[0].argmax() ]
print ("Age Range: "+ age)
print ("Predicted values of age are: \n", age_predicted)

cv2.putText(img=newimage, text=gender, org=(10, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),thickness=2)
cv2.putText(img=newimage, text=age, org=(30, 80), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),thickness=2)
cv2.imwrite("resulted.jpg", newimage)

#Displaying the image showing predicted age and gender on the image.
plt.subplot (121) , imshow (newimage)
plt.title ("Original Image")
plt.subplot (122) , imshow (new_img)
plt.title ("Resulted image")
plt.text (10, y+70, gender, fontsize=34)
plt.text (10, y+80, age, fontsize=34)
plt.show ()
#Main File

import cv2
import numpy as np, os, time,pickle
from ParseDocument import Document
from keras.models import load_model

counts = 0
global X,Y,W,H
#get the input image
imgPath = 'TestImages/Sample21.jpg'
image = cv2.imread(imgPath)
#
#image = cv2.resize(image,(1355,1045),cv2.INTER_AREA)
image = cv2.resize(image,(1360,1045),cv2.INTER_AREA)
img = image.copy()

obj = Document()
image = obj.processedImage(image)

image_with_lines = obj.dilateImage(image.copy(),60)

contours = obj.getCountours(image_with_lines.copy())
    #cv2.imwrite(baseDir +'/testDir/' + 'afterCnt1.jpg', image_With_Lines) 
contours = obj.sortCountours(contours, "top-to-bottom")

#getting the inmages line by line from each contours
for iter, line_Area in enumerate(contours):
    #print('Iteration:- ', iter)
    #print('len of Contours:- ', len(contours))
    counts = counts + 1
    x,y,w,h = cv2.boundingRect(line_Area)

    X,Y,W,H = x,y,w,h
    line_Image = image[y:y+h, x:x+w]
    #path = r'C:\Users\sachin\Desktop\Images\\' + str(counts)+'.png'
    
    line_Contours = obj.getCountours(image[y:y+h, x:x+w])
    line_Contours = obj.sortCountours(line_Contours,"left-to-right")
    #getting the Text 
    text = obj.getTextFromImage(image[y:y+h, x:x+w], line_Contours, Width=8, Height=8)   
    print(text)
    #break



    
"""
x,y,w,h = cv2.boundingRect(contours[4])       
cv2.imshow('img',image[y:y+h, x:x+w])
cv2.waitKey(0)


obj.getTextFromImage(image[y:y+h, x:x+w],line_Contours,Width=8, Height=8)


cv2.imwrite('email1.png', image[y:y+h, x:x+w])


#----------------------------------------------



_model = load_model('Models/Inference/_model.h5')

model = load_model('Models/Inference/CNN_OCRModel_v2.h5')

with open ('Models/Inference/CNNLabels.pkl','rb') as f:
    label = pickle.load(f)
label = dict([(v,k) for k,v in label.items()])


with open ('Models/Inference/_Labels.pkl','rb') as f:
    _label = pickle.load(f)
_label = dict([(v,k) for k,v in _label.items()])




cv2.imshow('thres',img)
cv2.waitKey(0)    




#testing work


img = cv2.imread('email1.png')



image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(image,(3,3),0)
thres = cv2.dilate(blur.copy(),np.ones((10,2), dtype='uint8'),iterations=1)
ret, thres = cv2.threshold(blur.copy(),0,255,cv2.THRESH_OTSU | cv2.THRESH_BINARY)
#close = cv2.morphologyEx(thres.copy(),cv2.MORPH_OPEN, np.zeros((5,5),dtype = 'uint8'))

#thres = cv2.adaptiveThreshold(blur,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,3,1 )
#erd = cv2.erode(thres,np.zeros((3,3),dtype = 'uint8'),iterations=3)

# compute gradients along the X and Y axis, respectively
gX = cv2.Sobel(thres, ddepth=cv2.CV_64F, dx=0, dy=1)
gY = cv2.Sobel(thres, ddepth=cv2.CV_64F, dx=1, dy=0)
 
# the `gX` and `gY` images are now of the floating point data type,
# so we need to take care to convert them back to an unsigned 8-bit
# integer representation so other OpenCV functions can utilize them
gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)
 
# combine the sobel X and Y representations into a single image
sobelCombined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
#final = cv2.dilate(sobelCombined,np.ones((6,10), dtype='uint8'))

dilate = cv2.dilate(thres.copy(),np.ones((8,2),np.uint8),iterations=1) 
cv2.imshow("Dilate", dilate)
cv2.waitKey(0)
cnt=0
_, contours, heir = cv2.findContours(thres.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
#cv2.drawContours(img,contours,-1,(0,0,255),1)
for cnt in contours:
    #getting the contour APPROXIMATION FOR "_" and '-' identification
    #epsilon = 0.1*cv2.arcLength(cnt,True)
    #approx = cv2.approxPolyDP(cnt,epsilon,True)
    #if len(approx) ==2:
    x,y,w,h = cv2.boundingRect(cnt)

    if w > 5 and h > 5:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.imshow('img',img)
        cv2.waitKey(0)
        #cropping the image
        res_img = thres[y:y+h, x:x+w]
        cv2.imwrite('./EvalImages/' + str(cnt) + '.png', res_img)
        cnt+=1
        res_img = thres[y-8:y+h+8, x-4:x+w+4]
        cv2.imwrite('./EvalImages/' + str(cnt) + '__.png', res_img)
        #res_img = cv2.erode(res_img, np.zeros((3,3),dtype='uint8'),iterations=1)
        res_img = cv2.resize(res_img,(32,32),cv2.INTER_AREA)
        #res_img = obj.getNewResizedImage(res_img,32)

        cv2.imshow('resize',res_img)
        cv2.waitKey(0)

        #prediction for _ 
        _prob = _model.predict(res_img.reshape(1,32,32,1))[0]
        print(_prob)
        index = np.argmax(_prob)
        print('Underscore found : ', _label[index])

        prob = model.predict(res_img.reshape(1,32,32,1))[0]
        index = np.argmax(prob)
        print('charcter is : ', label[index])
        #break


    #break

    


    #print(x,y,w,h)
    
    
    if (w <= 16 and w > 2) and (h <= 8) :
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.imshow('character', cv2.resize(img[y:y+h, x:x+w], (32,32), cv2.INTER_AREA))
        cv2.waitKey(0)
    cv2.imshow("Contours", img)
    cv2.waitKey(0)





# show our output images
cv2.imshow("image", img)
#cv2.imshow("erode", erd)
#cv2.imshow("thresholding", thres)



#cv2.imshow('close',close)
#cv2.imshow('erode',erd)
cv2.waitKey(0)
cv2.destroyAllWindows()



def getNewResizedImage(input_Image, image_Size):
    height,width = input_Image.shape
    #print (height, width)
    if width > height:
        aspect_Ratio = (float)(width/height)
        width = 20
        height = round(width/aspect_Ratio)
    else:
        aspect_Ratio = (float)(height/width)
        height = 20
        width = round(height/aspect_Ratio)
        
    input_Image = cv2.resize(input_Image, (width,height), interpolation = cv2.INTER_AREA )
    
    height,width = input_Image.shape
    
    number_Of_Column_To_Add = 32-width
    temp_Column = np.zeros( (height , int(number_Of_Column_To_Add/2)), dtype = np.uint8)
    input_Image = np.append(temp_Column, input_Image, axis=1)
    input_Image = np.append(input_Image, temp_Column, axis=1)


    height,width = input_Image.shape
    number_Of_Row_To_Add = 32-height
    temp_Row= np.zeros( (int(number_Of_Row_To_Add/2) , width ), dtype = np.uint8)
    input_Image = np.concatenate((temp_Row,input_Image))
    input_Image = np.concatenate((input_Image,temp_Row))

    return cv2.resize(input_Image, (image_Size,image_Size), interpolation = cv2.INTER_AREA )"""

"""



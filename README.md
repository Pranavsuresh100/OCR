# OCR
For hand written text identification (OCR)

To refer the Data set please use the Following Dataset :
**https://www.kaggle.com/vaibhao/handwritten-characters**

THe dataset contains 39 categories including :
1. Alphabtes (small and caps merged together just to avoid mis classification) -- > 26 
2. Digits (1 to 9) : Digit 0 is added to character O for avoiding misclassification
3. Some special characters which include &, #, $, @

**The data set contains Train and Validation folders containing 0.8+ Milion Training Records and 20,000 + validation records**
**ALl image are of 32,32 pixel black and white image..**

The arechitecture used to train the model is CNN along with deep learning network running paraller to CNN:
 Please refer the Flow diagram below :

![Model_Architecture](https://github.com/VaibhavKhamgaonkar/OCR/blob/master/modelStructure_CNN%2BLSTM.png)
 

* Adde Requirment files. Use 
**pip install -r /path/to/requirements.txt** for installation iof requirements.


**Description: -**
1. For Inference : 
* getText.py and ParseDocument_v2.py these 2 files sould be there in same folder as these files are used to identifying the Text from the scanned image document.
* configure the ParseDocument_v2.py file wiht the Model Path and label paths. The default location of all models and label files are inside Model folder. please change if you are storing these files elsewhere.
* Add the name of input scanned document in the getText.py file and save it. The sample files are stored in the SampleTestForms folders.
* Run getText.py file in command prompt ==> Program will fetch the hand writtern characters and print it on screen.


2. For Model Training:

* The Data should be present in in folder structure.

i.e.

     * ParentFolder 
     * |--Train ----
     * |        ----
     * |        ----
     * |
     * |--Validation ----
     * |             ----
     * |             ----


* Edit the CNN_mainScript.py file and configure the Training data path, Validation Data path, Number of classes, Learning rate, batchSize, regularization parameter.

* Edit the LSTM_Training.py file and update the Following parameters

1. Model checkpoint == Enter the name of model and path to which the Model get saved after each epochs if vac loss is reduced.
2. Label File Name
3. Update the Details al the bottom of the file.

**Note : the default location is in Current Working directory inside Models folder**



**Note**: _This OCR model is tuned for the sample forms added in the SampleTestForm folder. If you are trying for something different that the mentioned form then you have to tune it by yourself._


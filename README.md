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

![Parallel_Architecture](https://github.com/VaibhavKhamgaonkar/OCR/blob/master/modelStructure_CNN.png)
 

* Adde Requirment files. Use 
**pip install -r /path/to/requirements.txt** for installation iof requirements.




Note: Wait for further updates for infering the model to predict the hand written characters

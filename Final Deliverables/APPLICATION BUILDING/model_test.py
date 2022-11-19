from keras.models import load_model
import numpy as np
import cv2
from skimage.transform import resize

model=load_model('aslpng1.h5')
def detect(frame):
    img = resize(frame,(64,64,1))
    img = np.expand_dims(img,axis=0)
    if(np.max(img)>1):
        img=img/255.0
    prediction = model.predict(img)
    print(prediction)
    prediction =  np.argmax(model.predict(img), axis=-1)
    print(prediction)
    index=['A','B','C','D','E','F','G','H','I']
    print(index[int(prediction)])
frame=cv2.imread(r"C:\Users\USER\Desktop\IBM\test_set\A\103.png")
data=detect(frame)

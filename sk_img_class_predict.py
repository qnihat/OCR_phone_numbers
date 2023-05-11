import pickle
import numpy as np
import cv2
from skimage.transform import resize
import os

Categories = ["0","1","2","3","4","5","6","7","8","9","n"]
# Read byte from pickle model
test_model = pickle.load(open("weights/Classification_Model.p","rb"))

def predict_symbols(img_arr,coor):
    flat_data = []
    img_array = img_arr
    img_resized = resize(img_array, (150, 150, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    #print("Dimensions of original image are:", img_array.shape)

    y_output = test_model.predict(flat_data)
    y_output = Categories[y_output[0]]

    #print(f"PREDICTED OUTPUT IS: {y_output}, coor: {coor}")

    #cv2.imshow('img', img_resized)
    #cv2.waitKey(0)
    # cv2.imwrite('predicted_symbols/'+str(y_output)+'/'+file,img_array)
    return y_output
'''
for file in os.listdir('extracted_symbols/'):
    file_ = 'extracted_symbols/' + file
    predict_symbols(file_)
'''

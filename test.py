import cv2
import os
import numpy as np
# import tensorflow as tf
from keras.models import load_model
import joblib
import pandas as pd

#get path of current file
# BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
# BASE_FOLDER = 'ML'

# MODEL_PATHS ={
#     "heart" : os.path.join(BASE_FOLDER, "heart","ds1.joblib"),
#     "alz" : os.path.join(BASE_FOLDER,"alz","alzhimers_inceptionv3.hdf5"),
#     "diabetic" : os.path.join(BASE_FOLDER,"diabetic","model.hdf5"),
# }


# Define labels once
labels_dict = {
    "heart": ["Heartdisease Not Detected", "Heartdisease Detected"],
    "alz": ["Alzeihmers detected "  ,  "Alzeihmers Not detected"],
    "diabetic": ["Mild DR","Moderate DR","No DR","Severe DR"]
}

""" TEST FUNCTION FOR HEART DISEASE """

# def predict_heart():
#         df1=pd.read_csv('media/heart/test.csv')
#         df1.head()

#         model_heart=joblib.load("ML/heart/ds3.joblib")

#         pred=model_heart.predict(df1)
#         pred[0]

#         if pred[0]==0:
#             result='Heartdisease Not Detected'
#         elif pred[0]==1:
#             result='Heartdisease Detected'      
#         return result



def predict_heart():
        df1=pd.read_csv('media/heart/test.csv')
        df1.head()

        df=df1.iloc[:,:]

        model=joblib.load("ML/heart/ds2.joblib")
        pred=model.predict(df)
        print(pred)

        if pred==0:
            result='Heartdisease Not Detected'
        elif pred==1:
            result='Heartdisease Detected'      
         
        return result





""" TEST FUNCTION FOR ALZEIHMERS """

def predict_alz():
    
    model_alz = load_model('ML/alz/alzhimers_inceptionv3.hdf5')
    labels = labels_dict["alz"]  

    # Load and preprocess image
    img = cv2.imread("media/alz/test.jpg")
    if img is None:
        raise ValueError("Error loading image")
    
    img = cv2.resize(img, (299, 299))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Make predictions
    predictions = model_alz.predict(img, verbose=0)
    predicted_class_indices = np.argmax(predictions, axis=1)
    predicted_class = labels[predicted_class_indices[0]]

    return predicted_class 


""" TEST FUNCTION FOR DIABETIC RETINOPATHY"""

def predict_diabetic():
    
    model_diabetic = load_model('ML/diabetic/model.hdf5')
    labels = labels_dict["diabetic"]  

    # Load and preprocess image
    img = cv2.imread("media/diabetic/test.jpg")
    if img is None:
        raise ValueError("Error loading image")
    
    img = cv2.resize(img, (299, 299))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Make predictions
    predictions = model_diabetic.predict(img, verbose=0)
    predicted_class_indices = np.argmax(predictions, axis=1)
    predicted_class = labels[predicted_class_indices[0]]

    return predicted_class 


""" TEST FUNCTION FOR BONE CANCER"""

def predict_bone():
    
    model_diabetic = load_model('ML/diabetic/model.hdf5')
    labels = labels_dict["diabetic"]  

    # Load and preprocess image
    img = cv2.imread("media/diabetic/test.jpg")
    if img is None:
        raise ValueError("Error loading image")
    
    img = cv2.resize(img, (299, 299))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Make predictions
    predictions = model_diabetic.predict(img, verbose=0)
    predicted_class_indices = np.argmax(predictions, axis=1)
    predicted_class = labels[predicted_class_indices[0]]

    return predicted_class 
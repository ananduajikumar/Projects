import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

def predict():
                # Set the path to the test folder
                test_folder = 'media/input/test'

                # Load the trained model
                model_path = "ML/alzhimers_inceptionv3.hdf5"
                model = tf.keras.models.load_model(model_path)

                # Load and preprocess test images
                def load_and_preprocess_image(file_path):
                    img = cv2.imread(file_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
                    img = cv2.resize(img, (299, 299))  # Resize images to your model's input shape
                    img = img / 255.0  # Normalize pixel values
                    return np.expand_dims(img, axis=0)

                # Get a list of all image files in the test folder
                test_image_files = [os.path.join(test_folder, file) for file in os.listdir(test_folder) if file.endswith('.jpg')]

                # Load and preprocess each test image, then make predictions
                predictions = []
                for file_path in test_image_files:
                    test_image = load_and_preprocess_image(file_path)
                    prediction = model.predict(test_image)
                    label = 'non demented' if prediction > 0.9 else 'demented'
                    predictions.append((file_path, label))


    
                    if label=="demented":
                        
                        remedies="demented_remedies" 
                    else:
                        
                        remedies="NON-DEMENTED-remedies"
                
                    

                # Display the predictions
                #for file_path, label in predictions:
                    print(f"Image: {file_path}, Prediction: {label}")
                return(label,remedies)
predict()                
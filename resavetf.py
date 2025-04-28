from keras.models import load_model

# Load the existing model
model = load_model('alzhimers_inceptionv3.hdf5')

# Save in TensorFlow 1.13.1 compatible format
model.save('alzhimers_inceptionv3_tf1.h5')

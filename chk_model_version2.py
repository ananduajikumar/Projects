import h5py

# Replace 'your_model.h5' with the actual model file
model_path = "alzhimers_inceptionv3.hdf5"

with h5py.File(model_path, "r") as f:
    print(f.attrs["keras_version"])  # Keras version
    print(f.attrs["backend"])        # TensorFlow backend

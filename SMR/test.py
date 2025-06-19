import tensorflow as tf

# Check if TensorFlow is using the GPU
if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
else:
    print("GPU is not available")


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU

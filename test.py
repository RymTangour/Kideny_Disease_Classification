import tensorflow as tf
print(tf.__version__)  # Check TensorFlow version
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))  # Check GPU availability
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf  

# Check for GPU availability
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# Get GPU details
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print("GPU Name:", tf.config.experimental.get_device_details(gpus[0])['device_name'])

# Select device
device = "/GPU:0" if gpus else "/CPU:0"
print("Using device:", device)
import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4, 5, 6, 7'

import tensorflow as tf
from tensorflow.keras.applications import Xception
import numpy as np


from tensorflow.python.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
set_session(tf.compat.v1.Session(config=config))

num_samples = 10000
height = 214
width = 214
num_classes = 1000

#ms = tf.distribute.MirroredStrategy(devices=['/gpu:4', '/gpu:5', '/gpu:6', '/gpu:7',])
ms = tf.distribute.MirroredStrategy()
#print(ms.num_replicas_in_sync)
with ms.scope():
	model = Xception(weights=None,
					 input_shape=(height, width, 3),
					 classes=num_classes)
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# Instantiate the base model
# (here, we do it on CPU, which is optional).
#with tf.device('/cpu:0'):
#	 model = Xception(weights=None,
#					  input_shape=(height, width, 3),
#					  classes=num_classes)


# Replicates the model on 8 GPUs.
# This assumes that your machine has 8 available GPUs.
#parallel_model = multi_gpu_model(model, gpus=4)
#parallel_model.compile(loss='categorical_crossentropy',
#					   optimizer='rmsprop')

# Generate dummy data.
x = np.random.random((num_samples, height, width, 3))
y = np.random.random((num_samples, num_classes))

# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
#parallel_model.fit(x, y, epochs=20, batch_size=64)
model.fit(x, y, epochs=20, batch_size=64)

#train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
#
#def train(x, y):
#	model.train_on_batch(x, y)
#
#for x,y in train_dataset:
#	ms.run(train, args=(x, y,))


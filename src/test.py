import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import numpy as np
import tensorflow as tf

a = tf.constant([[1, 0, 3], [3, 0, 5]])
b = tf.constant([[1, 0, 1], [1, 0, 1]])
c = tf.reduce_sum(a, axis=-1, keepdims=True)
d = tf.math.count_nonzero(b, axis=-1, keepdims=True)


#print(pickle.HIGHEST_PROTOCOL)
#a = 1
## 保存
#with open('data.pickle', 'wb') as f:
#  pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
#  # 读取
#  with open('data.pickle', 'rb') as f:
#    b = pickle.load(f)

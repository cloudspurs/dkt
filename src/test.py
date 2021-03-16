#import os 
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import tensorflow as tf
#a = tf.constant([[1, 0, 3], [3, 0, 5]])
#b = tf.constant([[1, 0, 1], [1, 0, 1]])
#c = tf.reduce_sum(a, axis=-1, keepdims=True)
#d = tf.math.count_nonzero(b, axis=-1, keepdims=True)


# dataset.from_generator
#seqs = [0, 1, 2, 3, 4]
#x = lambda: seqs
#print(iter(x()))
#for b in iter(x()):
#	print(b)
#
#def gen():
#	for i in range(5):
#		yield i
#	
#print(iter(gen()))
#for b in iter(gen()):
#	print(b)
	
print(tf.__version__)
print("gpu", tf.test.is_gpu_available())

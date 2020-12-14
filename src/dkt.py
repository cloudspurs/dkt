### multi concepts dkt model

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
print('\nVisible GPU Devices:', os.environ['CUDA_VISIBLE_DEVICES'])

import tensorflow as tf
# 只能看到os.environ['CUDA_VISIBLE_DEVICES']设置的gpu
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# limit gpu memory
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import time, pickle, datetime

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Masking, Dense, LSTM, Dropout, TimeDistributed, concatenate, multiply
from tensorflow.keras.models import Model, Sequential


class Dkt():
	def __init__(self, num_skills, batch_size=100, hidden_units=200, optimizer='rmsprop', dropout_rate=0.5):
		self.__num_skills = num_skills
		self.__features = 2*num_skills
		self.__hidden_units = hidden_units
		self.mask_value = -1.0
		self.__batch_size = batch_size

		### model define
		x_t = Input(batch_shape=(None, None, 2*num_skills))
		mask = Masking(self.mask_value)(x_t)
		# stateful认为所有的输入在时序上是有关的，理论上不符合dkt每个学生为独立个体的情况
		lstm = LSTM(hidden_units, return_sequences=True)(mask)
		dropout = Dropout(dropout_rate)(lstm)
		y = TimeDistributed(Dense(num_skills, activation='sigmoid'))(dropout)

		q_tt = Input(batch_shape=(None, None, num_skills))
		mask_tt = Masking(self.mask_value)(q_tt)

		z = concatenate([y, mask_tt])
		#z = multiply([y, mask_tt])

		z = TimeDistributed(Dense(64, activation='sigmoid'))(z)
		z = TimeDistributed(Dense(1, activation='sigmoid'))(z)

		self.__model = Model(inputs=[x_t, q_tt], outputs=z)
		self.__model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.AUC(name='auc')])

	
	def train_and_test(self, seqs, epochs=1):

		dataset = self.get_data(seqs)
		
		### padded_batch之后，样本数量变成len(seqs)/batch_size, 然后按batch size划分训练，验证，测试集
		train_size = int(len(seqs) / self.__batch_size * 0.8) 
		val_size = int(train_size * 0.8)
		temp = dataset.take(train_size)
		train = temp.take(val_size)
		val = temp.skip(val_size)
		test = dataset.skip(train_size)

		# model weights file
		t = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y_%m_%d_%H_%M_%S')
		p = '../data/model/' + t + '_' + str(epochs) + 'epochs_' + str(self.__batch_size) + 'batch_size/'
		if not os.path.exists(p):
			os.mkdir(p)
		f = p + '{epoch:03d}epoch_{val_auc:.2f}val_auc_{val_loss:.2f}val_loss.h5'
		print('weights file:', f)

		#mc = ModelCheckpoint('../data/model/model_weights_' + f, monitor='auc', mode='max', save_weights_only=True, save_best_only=True)
		mc = ModelCheckpoint(f, save_weights_only=True)

		# train
		history = self.__model.fit(train, validation_data=val,
					epochs=epochs, batch_size=self.__batch_size, callbacks=[mc], verbose=1)
		print('loss:', history.history['loss'][-1])
		print('auc:', history.history['auc'][-1])
		print('val_loss:', history.history['val_loss'][-1])
		print('val_auc:', history.history['val_auc'][-1])

		# test
		result = self.__model.evaluate(test)
		print('Evaluate Result:', result)

	def predict(self, seqs):
		dataset = self.get_data(seqs)
		preds = self.__model.predict(dataset)
		print(preds.shape)
	
	def get_data(self, seqs):
		def gen_data():
			for seq in seqs:
				xs = []
				qtts = []
				ys = []
				for item in seq:
					skills = item[0]
					answer = item[1]

					x = np.array([-1]*8)
					for i,v in enumerate(skills):
						x[i] = answer * self.__num_skills + v
					qtt = np.array([-1]*8)
					for i,v in enumerate(skills):
						qtt[i] = v
					y = answer

					xs.append(x)
					qtts.append(qtt)
					ys.append(y)

				if len(xs) > 1: # answer more than one questions
					yield (xs[:-1], qtts[1:], ys[1:])

		# multi hot question skills
		@tf.autograph.experimental.do_not_convert
		def multi_hot(x, y, z):
			a = tf.one_hot(x, depth=2*self.__num_skills)
			b = tf.one_hot(y, depth=self.__num_skills)
			a = tf.reduce_sum(a, 1)
			b = tf.reduce_sum(b, 1)
			c = tf.expand_dims(z, -1)
			return (a, b, c)

		dataset = tf.data.Dataset.from_generator(gen_data, output_types=(tf.int32, tf.int32, tf.float32))
		dataset = dataset.map(multi_hot)
		dataset = dataset.padded_batch(self.__batch_size,
						#padded_shapes=([None, None], [None, None], [None, None]),
						padded_shapes=([None, 2*self.__num_skills], [None, self.__num_skills], [None, 1]),
						padding_values=(self.mask_value, self.mask_value, self.mask_value))
		# 把两个输入拼在一起
		dataset = dataset.map(lambda x, y, z: ((x, y), z))
		return dataset
	
	def save_weights(self, path='../data/model/dkt.h5'):
		self.__model.save_weights(path, overwrite=True)

	def load_weights(self, path='../data/model/model_weights.h5'):
		self.__model.load_weights(path)

	def save_lrp_weights(self, path='../data/model/dkt_lrp.weights'):
		h = self.__hidden_units

		lstm_weights = self.__model.layers[2].get_weights()
		
		# lstm weights order: i, f, c, o
		# lrp lstm weights order: i, c, f, o
		w_xh = lstm_weights[0].T
		w_hh = lstm_weights[1].T
		b_xh_h = lstm_weights[2]

		temp = np.copy(w_xh[h:2*h,:])
		w_xh[h:2*h,:] = w_xh[2*h:3*h,:]
		w_xh[2*h:3*h,:] = temp

		temp = np.copy(w_hh[h:2*h,:])
		w_hh[h:2*h,:] = w_hh[2*h:3*h,:]
		w_hh[2*h:3*h,:] = temp

		temp = np.copy(b_xh_h[h:2*h])
		b_xh_h[h:2*h] = b_xh_h[2*h:3*h]
		b_xh_h[2*h:3*h] = temp

		dense_weights = self.__model.layers[5].get_weights()
		w_hy = dense_weights[0].T
		b_hy = dense_weights[1]

		f = self.__model.layers[8].get_weights()
		f_w = f[0].T
		f_b = f[1]
		z = self.__model.layers[9].get_weights()
		z_w = z[0].T
		z_b = z[1]

		weights = {'w_xh': w_xh, 'w_hh': w_hh, 'b_xh_h': b_xh_h, 'w_hy': w_hy, 'b_hy': b_hy,
				'f_w': f_w, 'f_b': f_b, 'z_w': z_w, 'z_b': z_b}

		with open(path, mode='wb') as f:
			pickle.dump(weights, f)


### one question multi concepts dkt model (ednet dataset)

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
print('\nVisible GPU Device IDs:', os.environ['CUDA_VISIBLE_DEVICES'])

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')	# 只能看到os.environ['CUDA_VISIBLE_DEVICES']设置的gpu
print('Used GPU Devices:', len(gpus))
for gpu in gpus:	# limit gpu memory
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import pickle, datetime

from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Input, Masking, Dense, LSTM, Dropout, TimeDistributed, concatenate, multiply


class MultiDkt():
	def __init__(self, num_skills, batch_size=124, hidden_units=100, optimizer='adam', dropout_rate=0.5, strategy='min'):
		self.__num_skills = num_skills
		self.__features = 2*num_skills
		self.__hidden_units = hidden_units
		self.mask_value = -1.0
		self.__batch_size = batch_size

		def mean(y_pred, skills):
			y_pred = tf.reduce_sum(y_pred * skills, axis=-1, keepdims=True)
			non_zero = tf.math.count_nonzero(skills, axis=-1, keepdims=True)
			# 对padded的数据非零元素设置为1,防止出现0/0
			mask = tf.equal(non_zero, 0)
			non_zero = tf.where(mask, tf.cast(1, non_zero.dtype), non_zero)
			y_pred = y_pred / tf.cast(non_zero, y_pred.dtype)
			return y_pred

		def min(y_pred, skills):
			mask = tf.equal(skills, 1)
			y_pred = tf.where(mask, y_pred, 1)
			y_pred = tf.reduce_min(y_pred, axis=-1, keepdims=True)
			return y_pred

		def max(y_pred, skills):
			y_pred = tf.reduce_max(y_pred * skills, axis=-1, keepdims=True)
			return y_pred

		#self.get_pred = globals().get(strategy)
		#self.get_pred = globals()['__builtins__'][strategy]
		#self.get_pred = getattr(self, strategy)

		def get_target(y_true, y_pred):
			# 为了计算auc，将padded的-1变为0
			mask  = 1.0 - tf.cast(tf.equal(y_true, self.mask_value), y_true.dtype)
			y_true = y_true * mask

			# 切分知识点one-hot和标签
			skills, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)

			#y_pred = self.get_pred(y_pred, skills)
			y_pred = min(y_pred, skills)

			return y_true, y_pred

		def loss(y_true, y_pred):
			true, pred = get_target(y_true, y_pred)
			return binary_crossentropy(true, pred)

		class Auc(tf.keras.metrics.AUC):
			def update_state(self, y_true, y_pred, sample_weight=None):
				true, pred = get_target(y_true, y_pred)
				super(Auc, self).update_state(y_true=true, y_pred=pred, sample_weight=sample_weight)

		class BinaryAccuracy(tf.keras.metrics.BinaryAccuracy):
			def update_state(self, y_true, y_pred, sample_weight=None):
				true, pred = get_target(y_true, y_pred)
				super(BinaryAccuracy, self).update_state(y_true=true, y_pred=pred, sample_weight=sample_weight)
			
		x = Input(batch_shape=(None, None, 2*num_skills))
		mask = Masking(self.mask_value)(x)
		lstm = LSTM(hidden_units, return_sequences=True)(mask)
		dropout = Dropout(dropout_rate)(lstm)
		y = TimeDistributed(Dense(num_skills, activation='sigmoid'))(dropout)

		self.__model = Model(inputs=x, outputs=y)
		self.__model.compile(loss=loss, optimizer=optimizer, metrics=[Auc(name='auc'), BinaryAccuracy(name='acc')])
	

	def train_and_test(self, train_seqs, val_seqs, test_seqs, epochs=200):
		train = self.get_data(train_seqs)
		val = self.get_data(val_seqs)
		test = self.get_data(test_seqs)

		# model weights file
		#t = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y_%m_%d_%H_%M_%S')
		#p = '../data/model/' + t + '_' + str(epochs) + 'epochs_' + str(self.__batch_size) + 'batch_size/'

		p = '../data/model/ednet_200_no_delete_morethan_10/'
		if not os.path.exists(p):
		        os.mkdir(p)
		#f = p + '{epoch:03d}epoch_{val_auc:.5f}val_auc_{val_loss:.5f}val_loss.h5'
		f = p + 'dkt.h5'
		#print('weights file:', f)

		#mc = ModelCheckpoint(f, save_weights_only=True)
		mc = ModelCheckpoint(f, save_weights_only=True, save_best_only=True)

		# train
		history = self.__model.fit(train, validation_data=val, epochs=epochs, callbacks=[mc])
		#history = self.__model.fit(train, validation_data=val, epochs=epochs)

		print('loss:', history.history['loss'][-1])
		print('auc:', history.history['auc'][-1])
		print('val_loss:', history.history['val_loss'][-1])
		print('val_auc:', history.history['val_auc'][-1])

		# evaluate
		result = self.__model.evaluate(test)
		print('Evaluate loss, auc and acc:', result)
	

	def evaluate(self, seqs):
		dataset = self.get_data(seqs)
		metrics = self.__model.evaluate(dataset)
		print(metrics)


	def predict(self, seqs):
		dataset = self.get_data(seqs)
		preds = self.__model.predict(dataset)
		print(preds.shape)
	

	### last node predict accuracy
	def last_acc(self, seqs):
		dataset = self.get_data(seqs)

		y_true = []
		y_pred = []

		all_batch = int(len(seqs)/self.__batch_size)
		batch_num = 0

		for element in dataset:
			print('\rNow:', batch_num, 'All:', all_batch, 'Percentage:', batch_num/all_batch, end='')

			prods = self.__model.predict_on_batch(element)
			last_time_prods = prods[:,-1,:] # 获取每个序列最后时刻的输出

			s = batch_num * self.__batch_size
			e = s + self.__batch_size
			batch_seqs = seqs[s:e]

			batch_num = batch_num + 1

			for i, prob in enumerate(last_time_prods):
				seq = batch_seqs[i]
				last = seq[-1]
				skills = np.zeros((self.__num_skills,))
				for s_id in last[0]:
					skills[s_id] = 1
				pred = np.sum(skills * prob)
				pred = pred / len(last[0])
				pred = 1 if pred > 0.5 else 0

				y_true.append(last[1])
				y_pred.append(pred)
		
		acc = accuracy_score(y_true, y_pred)
		print('\nAccuracy', acc)
		return acc


	def get_right_or_error_predict_seqs(self, seqs):
		dataset = self.get_data(seqs)
		right_seqs = []
		error_seqs = []

		all_batch = int(len(seqs)/self.__batch_size)
		
		batch_num = 0
		for element in dataset:
			print('\rNow:', batch_num, 'All:', all_batch, 'Percentage:', batch_num/all_batch, end='')
			preds = self.__model.predict_on_batch(element)
			last_time_preds = preds[:,-1,:]
			#print(preds.shape)

			s = batch_num * self.__batch_size
			e = s + self.__batch_size
			batch_seqs = seqs[s:e]
			batch_num = batch_num + 1

			for i, prob in enumerate(last_time_preds):
				seq = batch_seqs[i]
				last = seq[-1]
				skills = np.zeros((self.__num_skills,))
				for s_id in last[0]:
					skills[s_id] = 1
				pred = np.sum(skills * prob)
				pred = pred / len(last[0])

				pred = 1 if pred > 0.5 else 0
				if pred == last[1]:
					right_seqs.append(seq)
				else:
					error_seqs.append(seq)

		print('\n', len(right_seqs))
		print(len(error_seqs))
		print(len(right_seqs) + len(error_seqs))

		with open('../data/lrp/right_seqs.bf', mode='wb') as f:
			pickle.dump(right_seqs, f)
		with open('../data/lrp/error_seqs.bf', mode='wb') as f:
			pickle.dump(error_seqs, f)

		return right_seqs, error_seqs


	def get_data(self, seqs):
		def gen_data():
			for seq in seqs:
				xs = []
				qtts = []
				ys = []
				for item in seq:
					skills = item[0]
					answer = item[1]

					x = np.array([-1]*7)
					for i,v in enumerate(skills):
						x[i] = answer*self.__num_skills+v
					#x[0] = answer * self.__num_skills+skills

					qtt = np.array([-1]*7)
					for i,v in enumerate(skills):
						qtt[i] = v
					#qtt[0] = skills

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
			### reduce_sum -> reduce_max 一个题目可能有重复的知识点 
			a = tf.reduce_max(a, 1)
			b = tf.reduce_max(b, 1)
			c = tf.expand_dims(z, -1)
			d = tf.concat([b, c], axis=-1)
			return (a, d)

		dataset = tf.data.Dataset.from_generator(gen_data, output_types=(tf.int32, tf.int32, tf.float32))
		dataset = dataset.map(multi_hot)
		dataset = dataset.padded_batch(self.__batch_size,
						padded_shapes=([None, None], [None, None]),
						padding_values=(self.mask_value, self.mask_value))
		return dataset

	
	def save_weights(self, path='../data/model/dkt.h5'):
		self.__model.save_weights(path, overwrite=True)

	def load_weights(self, path='../data/model/dkt.h5'):
		print(path)
		self.__model.load_weights(path)

	def save_lrp_weights(self, path='../data/model/lrp.weights'):
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

		dense_weights = self.__model.layers[4].get_weights()
		w_hy = dense_weights[0].T
		b_hy = dense_weights[1]

		weights = {'w_xh': w_xh, 'w_hh': w_hh, 'b_xh_h': b_xh_h, 'w_hy': w_hy, 'b_hy': b_hy}

		with open(path, mode='wb') as f:
			pickle.dump(weights, f)


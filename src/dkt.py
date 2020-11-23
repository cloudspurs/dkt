import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'

import tensorflow as tf

# limit gpu memory
#from keras.backend.tensorflow_backend import set_session
from tensorflow.python.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
set_session(tf.compat.v1.Session(config=config))


import time
import math
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, confusion_matrix

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Masking, Dense, TimeDistributed, Dropout, Input, concatenate, multiply
from keras.layers.recurrent import LSTM
from keras.utils import multi_gpu_model

from pad import pad_sequences
from one_hot import one_hot


class Dkt():
	def __init__(self, num_skills, batch_size=100, time_steps=50, hidden_units=50, optimizer='rmsprop', dropout_rate=0.5):
		self.__num_skills = num_skills
		self.__batch_size = batch_size
		self.__time_steps = time_steps
		self.__features = num_skills*2
		self.__hidden_units = hidden_units
		
		### origin dkt
		#def loss(y_true, y_pred):
		#	labels = y_true[:,:,num_skills]
		#	skills = y_true[:,:,0:num_skills]
		#	pred_labels = K.sum(y_pred * skills, axis=2)
		#	return K.binary_crossentropy(labels, pred_labels)

		### one question multi concepts dkt
		def loss(y_true, y_pred):
			labels = y_true[:,:,num_skills]
			y_pred = tf.reshape(y_pred, (batch_size, -1))
			return K.binary_crossentropy(tf.cast(labels, tf.float32), y_pred)

		### sequential model
		#self.__model = Sequential()
		#self.__model.add(Masking(-1.0, batch_input_shape=(batch_size, None, self.__features)))
		#self.__model.add(LSTM(hidden_units, return_sequences=True, stateful=True))
		#self.__model.add(Dropout(dropout_rate))
		#self.__model.add(TimeDistributed(Dense(num_skills, activation='sigmoid')))
		#self.__model.add(TimeDistributed(Dense(32, activation='sigmoid')))
		#self.__model.add(TimeDistributed(Dense(1, activation='sigmoid')))
		#self.__model.summary()
		#self.__model.compile(loss=loss, optimizer=optimizer)

		### functional model
		self.ms = tf.distribute.MirroredStrategy()
		with self.ms.scope():
			x_t = Input(batch_shape=(None, None, self.__features))
			mask = Masking(-1.0)(x_t)
			# MirroredStrategy not support stateful=True
			#lstm = LSTM(hidden_units, return_sequences=True, stateful=True)(mask)
			lstm = LSTM(hidden_units, return_sequences=True)(mask)
			dropout = Dropout(dropout_rate)(lstm)
			y = TimeDistributed(Dense(num_skills, activation='sigmoid'))(dropout)

			x_tt = Input(batch_shape=(None, None, num_skills))
			mask_tt = Masking(-1.0)(x_tt)
			z = concatenate([y, mask_tt])
			#z = multiply([y, mask_tt])
			z = TimeDistributed(Dense(32, activation='sigmoid'))(z)
			z = TimeDistributed(Dense(1, activation='sigmoid'))(z)
			self.__model = Model(inputs=[x_t, x_tt], outputs=z)
			#self.__model = Model(inputs=x_t, outputs=z)
			#self.__model.compile(loss=loss, optimizer=optimizer)
			self.__model.compile(loss='binary_crossentropy', optimizer=optimizer)
		#self.__model.summary()

	def train_and_test(self, path):
		df = pd.read_csv(path)
		seqs = df.groupby('stu_docu_id').apply(
			lambda r: (
				r['correctness'].values,
				r['correctness'].values,
				r['correctness'].values
			)
		)

		print(type(seqs))
		print(seqs.keys())
		print(type(seqs.get('u103390.csv')))
		print(type(seqs.get('u103390.csv')))
		print(seqs.get('u103390.csv'))

		dataset = tf.data.Dataset.from_generator(lambda: seqs, output_types=(tf.int32, tf.int32, tf.float32))
		dataset = dataset.map(lambda x, y, z: (
				tf.one_hot(x, depth=188*2),
				#tf.one_hot(y, depth=188),
				tf.expand_dims(z, -1)
				#z
		))
		dataset = dataset.padded_batch(1000, padded_shapes=([None, None], [None, None]),
										padding_values=(-1.0, -1.0), drop_remainder=True)
		self.__model.fit(dataset, epochs=1200, batch_size=1000)

	
	def ttrain_and_test(self, train_seqs, test_seqs, epochs=1):
		seqs = []
		for seq in train_seqs:
			xs = []
			qtts = []
			ys = []
			for item in seq:
				skills = item[0]
				answer = item[1]

				x = np.array([-1]*10)
				for i,v in enumerate(skills):
					x[i] = answer*self.__num_skills+v
				qtt = np.array([-1]*10)
				for i,v in enumerate(skills):
					qtt[i] = v
				#x = [item[1]*self.__num_skills+x for x in item[0]]
				#qtt = item[0]
				y = answer

				#x = item[1]*self.__num_skills+item[0][0]
				#qtt = item[0][0]
				#y = item[1]

				xs.append(x)
				qtts.append(qtt)
				ys.append(y)
			seqs.append((xs, qtts, ys))

		def g():
			for i in seqs:
				yield i
		gg = g()
		print(type(next(gg)))
		print(type(next(gg)[0]))
		print(type(next(gg)[0][0]))

		def mmm(x, y, z):
			a = tf.one_hot(x, depth=2*self.__num_skills)
			a = tf.reduce_sum(a, 1)
			b = tf.one_hot(y, depth=self.__num_skills)
			b = tf.reduce_sum(b, 1)
			c = tf.expand_dims(z, -1)
			return (a, b, c)

		dataset = tf.data.Dataset.from_generator(lambda: seqs, output_types=(tf.int32, tf.int32, tf.float32))
		#dataset = tf.data.Dataset.from_generator(lambda: seqs, output_types=(tf.int32, tf.int32, tf.float32),
		#			output_shapes=([None, None], [None, None], [None]))

		#dataset = dataset.map(lambda x, y, z: (
		#		tf.one_hot(x, depth=2*self.__num_skills),
		#		tf.one_hot(y, depth=self.__num_skills),
		#		#tf.one_hot(x, depth=2*self.__num_skills),
		#		#tf.one_hot(y, depth=self.__num_skills),
		#		tf.expand_dims(z, -1)
		#))
		dataset = dataset.map(mmm)

		#dataset = dataset.padded_batch(self.__batch_size, padded_shapes=([None, None], [None, None]),
		#								padding_values=(-1.0, -1.0), drop_remainder=True)
		dataset = dataset.padded_batch(self.__batch_size, padded_shapes=([None, None], [None, None], [None, None]),
										padding_values=(-1.0, -1.0, -1.0), drop_remainder=True)

		# multi inputs (x, y), one output z
		dataset = dataset.map(lambda x, y, z: (
				(x, y),
				z
		))

		self.__model.fit(dataset, epochs=100, batch_size=self.__batch_size)

	
#	def train_and_test(self, train_seqs, test_seqs, epochs=1):
#		assert(epochs > 0)	  
#
#		self.print_answer_nums(train_seqs, test_seqs)
#		
#		aucs = np.zeros(epochs) 
#		max_auc = 0.0
#
#		for e in range(epochs):
#			print('\nBatch: ', e)
#			start = time.time()
#
#			#self.train(train_seqs)
#			#labels, pred_labels = self.predict(test_seqs)
#			self.ms.run(self.train, args=(train_seqs,))
#
#			#auc, acc, pre = self.evaluates(labels, pred_labels)
#			#aucs[e] = auc
#
#			#end = time.time()
#			#print('One Epoch Time: ', end - start)
#
#			#if auc > max_auc:
#			#	max_auc =  auc
#			#	self.save_weights()
#			#	#self.save_lstm_weights()
#
#		print('\nMax Auc: ', max_auc)


	def train(self, sequences):
		loss = 0.0 
		answers = 0

		for start in range(0, len(sequences), self.__batch_size):
			print('\rNow:', int(start/self.__batch_size), 'All:', int(len(sequences)/self.__batch_size), end='', flush=True)
			answer, features, xtt, labels,  = self.__get_next_batch(sequences, start)
			answers += answer
			batch_loss = self.__model.train_on_batch([features, xtt], labels)
			self.__model.reset_states()

			loss += batch_loss

		print('Loss: ', loss)
		print('Real Train Answers: ', answers)


	def get_correct_predict_seqs(self, sequences, threshold=0.5):
		correct_seqs = []
		
		for start in range(0, len(sequences), self.__batch_size):
			end = min(len(sequences), start+self.__batch_size)
			now_seqs = sequences[start:end]

			answer, features, xtt, labels = self.__get_next_batch(sequences, start)
			self.__model.reset_states()
			pred_labels = self.__model.predict_on_batch(features)

			pred_labels = np.squeeze(np.array(pred_labels))

			skill = labels[:,-1,0:self.__num_skills]
			real_pred_labels  = np.sum(pred_labels[:,-1,:] * skill, axis=1)
			binary_preds = [1 if p > threshold else 0 for p in real_pred_labels]

			real_labels = labels[:,-1,self.__num_skills]

			for i, l in enumerate(real_labels):
				if l != -1:
					if binary_preds[i] == l:
						correct_seqs.append(now_seqs[i])
		
		return correct_seqs


	def predict_last_node(self, sequences):
		flat_labels = []
		flat_pred_labels = []

		answers = 0

		for start in range(0, len(sequences), self.__batch_size):
			answer, features, xtt, labels = self.__get_next_batch(sequences, start)
			answers += answer
			pred_labels = self.__model.predict_on_batch(features)

			self.__model.reset_states()

			#(batch_size, None, num_skills)
			pred_labels = np.squeeze(np.array(pred_labels))
			#(batch_size)
			real_labels = labels[:,-1,self.__num_skills]

			# which skill is answered
			#(batch_size, num_skills)
			skill = labels[:,-1,0:self.__num_skills]
			#(batch_size)
			real_pred_labels  = np.sum(pred_labels[:,-1,:] * skill, axis=1)

			flat_real_labels = np.reshape(real_labels, [-1])
			flat_real_pred_labels = np.reshape(real_pred_labels, [-1])

			mask_index = np.where(flat_real_labels == -1.0)[0]
			flat_real_labels = np.delete(flat_real_labels, mask_index)
			flat_real_pred_labels = np.delete(flat_real_pred_labels, mask_index)
			flat_labels.extend(flat_real_labels)
			flat_pred_labels.extend(flat_real_pred_labels)

		print('Real Test Answers: ', answers)
		return flat_labels, flat_pred_labels

	
	def predict(self, sequences):
		flat_labels = []
		flat_pred_labels = []

		answers = 0

		for start in range(0, len(sequences), self.__batch_size):
			answer, features, xtt, labels = self.__get_next_batch(sequences, start)
			answers += answer
			pred_labels = self.__model.predict_on_batch([features, xtt])
			self.__model.reset_states()

			pred_labels = np.squeeze(np.array(pred_labels))
			real_labels = labels[:,:,self.__num_skills]

			# which skill is answered
			#skill = labels[:,:,0:self.__num_skills]
			#real_pred_labels  = np.sum(pred_labels * skill, axis=2)

			real_pred_labels = pred_labels

			flat_real_labels = np.reshape(real_labels, [-1])
			flat_real_pred_labels = np.reshape(real_pred_labels, [-1])

			mask_index = np.where(flat_real_labels == -1.0)[0]
			flat_real_labels = np.delete(flat_real_labels, mask_index)
			flat_real_pred_labels = np.delete(flat_real_pred_labels, mask_index)
			flat_labels.extend(flat_real_labels)
			flat_pred_labels.extend(flat_real_pred_labels)

		print('Real Test Answers: ', answers)
		return flat_labels, flat_pred_labels


	def evaluates(self, labels, preds, threshold=0.5):
		binary_preds = [1 if p > threshold else 0 for p in preds]
		auc = roc_auc_score(labels, preds)
		acc = accuracy_score(labels, binary_preds)
		pre = precision_score(labels, binary_preds)
		matrix = confusion_matrix(labels, binary_preds)


		print('\nAuc: ', auc)
		print('Acc: ', acc)
		print('Pre: ', pre)
		print('Matrix: ', matrix)
		return auc, acc, pre


	def __get_next_batch(self, seqs, start):
		end = min(len(seqs), start+self.__batch_size)
		x = []
		x_tt = []
		y = []

		for seq in seqs[start:end]:
			features = []
			xtts = []
			labels = []

			feature = np.zeros(self.__num_skills*2)
			#last_skill_id = seq[-1][0]

			for skill_ids, correct_or_wrong in seq:
				features.append(feature)
				feature = np.zeros(self.__num_skills*2)
				for skill_id in skill_ids:
					feature[correct_or_wrong * self.__num_skills + skill_id] = 1

				label = np.zeros(self.__num_skills+1)
				label[skill_id] = 1
				label[-1] = correct_or_wrong
				labels.append(label)


				xtt = np.zeros(self.__num_skills)
				for skill_id in skill_ids:
					xtt[skill_id] = 1
				xtts.append(xtt)

				#if skill_id > 110:
				#	feature = np.zeros(self.__num_skills*2)
				#	label = np.zeros(self.__num_skills+1)
				#	label[skill_id-111] = 1
				#	label[self.__num_skills] = correct_or_wrong
				#else:
				#	feature, label = one_hot(skill_id, correct_or_wrong, self.__num_skills)
				#labels.append(label)

			x.append(features)
			x_tt.append(xtts)
			y.append(labels)

		max_length = max([len(s) for s in x]) 

		# fill up sequences to batch size
		if len(x) < self.__batch_size:
			for e in range(self.__batch_size - len(x)):
				x.append([np.array([-1.0 for i in range(0, self.__num_skills*2)])])
				x_tt.append([np.array([-1.0 for i in range(0, self.__num_skills)])])
				y.append([np.array([0.0 for i in range(0, self.__num_skills+1)])])

		# pad seqs to the same size max_length
		x = pad_sequences(x, padding='post', maxlen = max_length, dim=self.__num_skills*2, value=-1.0)
		x_tt = pad_sequences(x_tt, padding='post', maxlen = max_length, dim=self.__num_skills, value=-1.0)
		y = pad_sequences(y, padding='post', maxlen = max_length, dim=self.__num_skills+1, value=-1.0)
			
		return max_length*self.__batch_size, x, x_tt, y 


	def save_lstm_weights(self, path='../data/model/dkt_lrp.weights'):
		h = self.__hidden_units

		lstm_weights = self.model.layers[2].get_weights()
		
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

		dense_weights = self.model.layers[5].get_weights()
		w_hy = dense_weights[0].T
		b_hy = dense_weights[1]

		f = self.model.layers[8].get_weights()
		f_w = f[0].T
		f_b = f[1]
		z = self.model.layers[9].get_weights()
		z_w = z[0].T
		z_b = z[1]

		weights = {'w_xh': w_xh, 'w_hh': w_hh, 'b_xh_h': b_xh_h, 'w_hy': w_hy, 'b_hy': b_hy,
				'f_w': f_w, 'f_b': f_b, 'z_w': z_w, 'z_b': z_b}

		with open(path, mode='wb') as f:
			pickle.dump(weights, f)


#	def save_lstm_weights(self, path='model/lstm_weights'):
#		h = self.__hidden_units
#
#		lstm_weights = self.__model.layers[1].get_weights()
#		#dense_weights = self.__model.layers[3].get_weights()
#		dense_weights = self.__model.layers[2].get_weights()
#		
#		# lstm weights order: i, f, c, o
#		# lrp lstm weights order: i, c, f, o
#		w_xh = lstm_weights[0].T
#		w_hh = lstm_weights[1].T
#		b_xh_h = lstm_weights[2]
#
#		temp = np.copy(w_xh[h:2*h,:])
#		w_xh[h:2*h,:] = w_xh[2*h:3*h,:]
#		w_xh[2*h:3*h,:] = temp
#
#		temp = np.copy(w_hh[h:2*h,:])
#		w_hh[h:2*h,:] = w_hh[2*h:3*h,:]
#		w_hh[2*h:3*h,:] = temp
#
#		temp = np.copy(b_xh_h[h:2*h])
#		b_xh_h[h:2*h] = b_xh_h[2*h:3*h]
#		b_xh_h[2*h:3*h] = temp
#
#		w_hy = dense_weights[0].T
#		b_hy = dense_weights[1]
#
#		weights = {'w_xh': w_xh, 'w_hh': w_hh, 'b_xh_h': b_xh_h, 'w_hy': w_hy, 'b_hy': b_hy}
#
#		with open(path, mode='wb') as f:
#			pickle.dump(weights, f)

	
	def save_weights(self, path='../data/model/dkt.h5'):
		self.__model.save_weights(path, overwrite=True)


	def load_weights(self, path='../data/model/dkt.h5'):
		self.__model.load_weights(path)


	def print_answer_nums(self, train, test):
		num_train = 0
		num_test = 0

		for seq in train:
			num_train += len(seq)

		for seq in test:
			num_test += len(seq) 

		print('\nAll Answers: ', num_train + num_test)
		print('Train Answers: ', num_train)
		print('Test Answers: ', num_test)


	def print_weights(self):
		print('\nModel Summary')
		self.__model.summary()
		print('\nModel Weights')
		for e in zip(self.__model.layers[1].trainable_weights, self.__model.layers[1].get_weights()):
			print('\t%s: %s' % (e[0],e[1].shape))

		for e in zip(self.__model.layers[2].trainable_weights, self.__model.layers[2].get_weights()):
			print('\t%s: %s' % (e[0],e[1].shape))

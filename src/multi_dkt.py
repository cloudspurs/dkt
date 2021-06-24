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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Input, Masking, Dense, LSTM, Dropout, TimeDistributed, concatenate, multiply


class MultiDkt():
	#def __init__(self, num_skills, batch_size=124, hidden_units=100, optimizer='adam', dropout_rate=0.5, strategy='min'):
	def __init__(self, args):
		self.__num_skills = args.concepts
		self.__hidden_units = args.lstm_dim 
		self.mask_value = args.mask_value
		self.__batch_size = args.batch_size

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
			if args.loss == 'min':
				y_pred = min(y_pred, skills)
			if args.loss == 'max':
				y_pred = max(y_pred, skills)
			if args.loss == 'mean':
				y_pred = mean(y_pred, skills)

			return y_true, y_pred

		def loss(y_true, y_pred):
			true, pred = get_target(y_true, y_pred)
			return binary_crossentropy(true, pred)

		def rmse(y_true, y_pred):
			true, pred = get_target(y_true, y_pred)
			return tf.math.sqrt(tf.reduce_mean(tf.math.square(pred-true), axis=-1, keepdims=True))

		class Auc(tf.keras.metrics.AUC):
			def update_state(self, y_true, y_pred, sample_weight=None):
				true, pred = get_target(y_true, y_pred)
				super(Auc, self).update_state(y_true=true, y_pred=pred, sample_weight=sample_weight)

		class BinaryAccuracy(tf.keras.metrics.BinaryAccuracy):
			def update_state(self, y_true, y_pred, sample_weight=None):
				true, pred = get_target(y_true, y_pred)
				super(BinaryAccuracy, self).update_state(y_true=true, y_pred=pred, sample_weight=sample_weight)
			
		x = Input(batch_shape=(None, None, 2*self.__num_skills))
		mask = Masking(self.mask_value)(x)
		lstm = LSTM(self.__hidden_units, return_sequences=True)(mask)
		dropout = Dropout(args.dropout_rate)(lstm)
		y = TimeDistributed(Dense(self.__num_skills, activation='sigmoid'))(dropout)

		self.__model = Model(inputs=x, outputs=y)
		self.__model.compile(loss=loss, optimizer=args.optimizer, metrics=[Auc(name='auc'), BinaryAccuracy(name='acc'), rmse])
	

	#def train_and_test(self, train_seqs, val_seqs, test_seqs, args):
	#	train = self.get_data(train_seqs)
	#	val = self.get_data(val_seqs)
	#	test = self.get_data(test_seqs)

	def train_and_test(self, train, val, test, args):
		train = self.get_data_new(train)
		val = self.get_data_new(val)
		test = self.get_data_new(test)

		# model weights file
		p = '../data/model/ednet_200_no_delete_morethan_10/'
		if not os.path.exists(p):
		        os.mkdir(p)
		f = p + args.loss + '_new_multi_dkt.h5'

		mc = ModelCheckpoint(f, save_weights_only=True, save_best_only=True)
		es = EarlyStopping(monitor='val_loss', patience=10)

		# train
		history = self.__model.fit(train, validation_data=val, epochs=args.epochs, batch_size=self.__batch_size, callbacks=[mc, es])

		# evaluate
		self.__model.load_weights(f)
		result = self.__model.evaluate(test)
		return result
	

	def evaluate(self, seqs):
		dataset = self.get_data(seqs)
		metrics = self.__model.evaluate(dataset)
		print(metrics)


	def predict(self, seqs):
		dataset = self.get_data(seqs)
		preds = self.__model.predict(dataset)
		print(preds.shape)
	

	def get_data_new(self, path):
		def gen_data():
			with open(path, mode='rt') as f:
				questions = []
				answers = []
				xs = []
				qtts = []
				ys = []
				for index, line in enumerate(f):
					if index % 2 == 0:
						t = line.split(',')
						for e in t:
							skills = e.split('_')
							questions.append([int(s) for s in skills])

					if index % 2 == 1:
						answers = [int(e) for e in line.split(',')]

						for i, a in enumerate(answers):
							x = np.array([-1]*7)
							for j,q in enumerate(questions[i]):
								x[j] = a*self.__num_skills+q
							xs.append(x)

							qtt = np.array([-1]*7)
							for j,q in enumerate(questions[i]):
								qtt[j] = q
							qtts.append(qtt)

							ys.append(a)

						if len(xs) > 1: # answer more than one questions
							yield (xs[:-1], qtts[1:], ys[1:])
						questions = []
						answers = []
						xs = []
						qtts = []
						ys = []


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


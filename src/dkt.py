import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3, 4, 5, 6, 7'
print('\nVisible GPU Devices:', os.environ['CUDA_VISIBLE_DEVICES'])

# limit gpu memory
import tensorflow as tf
# 只能看到os.environ['CUDA_VISIBLE_DEVICES']设置的gpu
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import time, pickle, datetime
import numpy as np

#from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, confusion_matrix

#from keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Masking, Dense, LSTM, Dropout, TimeDistributed, concatenate, multiply

from pad import pad_sequences
from one_hot import one_hot


class Dkt():
	def __init__(self, num_skills, batch_size=100, hidden_units=50, optimizer='rmsprop', dropout_rate=0.5):
		self.__num_skills = num_skills
		self.__features = num_skills*2
		self.__hidden_units = hidden_units
		self.mask_value = -1.0

		#ms = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1'])
		ms = tf.distribute.MirroredStrategy()
		print('\nUsed GPU Devices:', ms.num_replicas_in_sync)
		self.__batch_size = batch_size * ms.num_replicas_in_sync
		print('Batch Size:', self.__batch_size)

		with ms.scope():
			x_t = Input(batch_shape=(None, None, 2*num_skills))
			mask = Masking(self.mask_value)(x_t)
			# MirroredStrategy not support stateful=True
			#lstm = LSTM(hidden_units, return_sequences=True, stateful=True)(mask)
			lstm = LSTM(hidden_units, return_sequences=True)(mask)
			dropout = Dropout(dropout_rate)(lstm)
			y = TimeDistributed(Dense(num_skills, activation='sigmoid'))(dropout)

			q_tt = Input(batch_shape=(None, None, num_skills))
			mask_tt = Masking(self.mask_value)(q_tt)

			z = concatenate([y, mask_tt])
			#z = multiply([y, mask_tt])

			z = TimeDistributed(Dense(32, activation='sigmoid'))(z)
			z = TimeDistributed(Dense(1, activation='sigmoid'))(z)

			self.__model = Model(inputs=[x_t, q_tt], outputs=z)
			self.__model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.AUC(name='auc')])
		#self.__model.summary()
	
	
	def train_and_test(self, seqs, epochs=200):
		print('\nEpochs:', epochs)

		### 根据每个学生作答序列生成输入输出
		def gen_data():
			for seq in seqs:
				xs = []
				qtts = []
				ys = []
				for item in seq:
					skills = item[0]
					answer = item[1]

					'''
						每个题目对应多个知识点，设置10维数组存放题目ID，
						因为转成tensor需要固定长度（接下来优化）
					'''
					# tf 2.4.0
					#x = []
					#for i,v in enumerate(skills):
					#	x.append(answer*self.__num_skills+v)
					#qtt = [] 
					#for i,v in enumerate(skills):
					#	qtt.append(v)

					x = np.array([-1]*8)
					for i,v in enumerate(skills):
						x[i] = answer*self.__num_skills+v
					qtt = np.array([-1]*8)
					for i,v in enumerate(skills):
						qtt[i] = v
					y = answer

					xs.append(x)
					qtts.append(qtt)
					ys.append(y)
				if len(xs) > 1: # answer more than one questions
					yield (xs[:-1], qtts[1:], ys[1:])
					# tf 2.4.0
					#yield tf.ragged.constant(xs[:-1]), tf.ragged.constant(qtts[1:]), ys[1:]

		# multi hot question skills
		def multi_hot(x, y, z):
			a = tf.one_hot(x, depth=2*self.__num_skills)
			b = tf.one_hot(y, depth=self.__num_skills)
			a = tf.reduce_sum(a, 1)
			b = tf.reduce_sum(b, 1)
			c = tf.expand_dims(z, -1)
			return (a, b, c)

		dataset = tf.data.Dataset.from_generator(gen_data, output_types=(tf.int32, tf.int32, tf.float32))
		# tf 2.4.0
		#dataset = tf.data.Dataset.from_generator(gen_data,
		#				output_signature=(
		#					tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32),
		#					tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32),
		#					tf.TensorSpec(shape=(None,), dtype=tf.float32)))

		# shape: (None, 2*188) (None, 188), (None, 1)
		dataset = dataset.map(multi_hot)
		# shape: (batch_size, None, 2*188) (batch_size, None, 188) (batch_size, None, 1)
		dataset = dataset.padded_batch(self.__batch_size,
						padded_shapes=([None, None], [None, None], [None, None]),
						padding_values=(self.mask_value, self.mask_value, self.mask_value,),
						drop_remainder=True)
		# 把两个输入拼在一起
		dataset = dataset.map(lambda x, y, z: ((x, y), z))
		
		### padded_batch之后，样本数量变len(seqs)/batch_size, 然后按batch size划分训练，验证，测试集
		train_size = int(len(seqs) / self.__batch_size * 0.8) 
		val_size = int(train_size * 0.8)
		temp = dataset.take(train_size)
		train = temp.take(val_size)
		val = temp.skip(val_size)
		test = dataset.skip(train_size)

		t = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('_%Y_%m_%d_%H_%M_%S')
		f = str(epochs) + 'epochs_' + str(self.__batch_size) + 'batch_size' + t + '.h5'
		print('weights file:', f)
		mc = ModelCheckpoint('../data/model/model_weights_' + f,
				monitor='auc', mode='max', save_weights_only=True, save_best_only=True)

		history = self.__model.fit(train, validation_data=train,
					epochs=epochs, batch_size=self.__batch_size, callbacks=[mc], verbose=1)

		with open('../data/model/history.bf', mode='wb') as f:
			pickle.dump(history.history, f)
		print('loss:', history.history['loss'][-1])
		print('auc:', history.history['auc'][-1])
		print('val_loss:', history.history['val_loss'][-1])
		print('val_auc:', history.history['val_auc'][-1])

		result = self.__model.evaluate(test)
		with open('../data/model/test_loss_auc.bf', mode='wb') as f:
			pickle.dump(result, f)
		print('Test loss and auc:', result)

	
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


	#def train(self, sequences):
	#	loss = 0.0 
	#	answers = 0

	#	for start in range(0, len(sequences), self.__batch_size):
	#		print('\rNow:', int(start/self.__batch_size), 'All:', int(len(sequences)/self.__batch_size), end='', flush=True)
	#		answer, features, xtt, labels,  = self.__get_next_batch(sequences, start)
	#		answers += answer
	#		batch_loss = self.__model.train_on_batch([features, xtt], labels)
	#		self.__model.reset_states()

	#		loss += batch_loss

	#	print('Loss: ', loss)
	#	print('Real Train Answers: ', answers)


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


import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time, pickle, argparse, datetime, os, ast
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import binary_crossentropy

from dkt import Dkt
from akt_data.data import load_data

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Dkt model train and test')

	parser.add_argument("--train", default=False, action='store_true', help='train or test the model')
	parser.add_argument("--test", default=False, action='store_true', help='test the model')
	parser.add_argument("--cross_validation", default=False, action='store_true', help='k-fold cross validation')

	parser.add_argument('--gpus', type=str, default='0', help='want to using gpus, e.g "python main.py --gpus=0,1,2,3"')

	parser.add_argument('--epochs', type=int, default=300)
	parser.add_argument('--batch_size', type=int, default=24)
	parser.add_argument('--lstm_dim', type=int, default=128)
	parser.add_argument('--dropout_rate', type=float, default=0.5)
	parser.add_argument('--mask_value', type=float, default=0.0)

	parser.add_argument('--dataset', type=str, default='assist2009_pid')

	parser.add_argument('--seqlen', type=int, default=200)

	args = parser.parse_args()

	if(args.train is False and args.test is False and args.cross_validation is False):
		print('Please input "python main.py --train" or "python main.py --test" or "python main.py --cross_validation"')
		exit()
	
	datasets = ['statics', 'assist2009_pid', 'assist2015', 'assist2017_pid']
	dataset = args.dataset
	if dataset not in datasets:
		print('Dataset', dataset, 'not exist!')
		exit()
	concepts = [1223, 110, 100, 102]
	questions = [0, 16891, 0, 3162]
	
	if dataset == 'statics':
		args.concepts = concepts[0]
	if dataset == 'assist2009_pid':
		args.concepts = concepts[1]
		args.questions = questions[1]
	if dataset == 'assist2015':
		args.concepts = concepts[2]
	if dataset == 'assist2017_pid':
		args.concepts = concepts[3]
		args.questions = questions[3]
	print(args)

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
	gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
	print('\nCUDA_VISIBLE_DEVICES Environment:', os.environ['CUDA_VISIBLE_DEVICES'])
	print('Used GPU Devices Number:', len(gpus))

	# limit gpu memory
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)

	train_data_dir = '../data/akt_data/' + args.dataset + '/' + args.dataset + '_train1.csv'
	test_data_dir = '../data/akt_data/' + args.dataset + '/' + args.dataset + '_test1.csv'
	valid_data_dir = '../data/akt_data/' + args.dataset + '/' + args.dataset + '_valid1.csv'
	train_seqs	= load_data(train_data_dir, 4)
	test_seqs	= load_data(test_data_dir, 4)
	valid_seqs	= load_data(valid_data_dir, 4)
	print(len(train_seqs))
	exit()

#	with open(args.data, mode='rb') as f:
#		seqs = pickle.load(f)
#
#	split = int(0.8 * len(seqs))
#	train_seqs = seqs[:split]
#	val_seqs = seqs[int(0.8*split):split]
#	test_seqs = seqs[split:]

	#def get_target(y_true, y_pred):
	#	skills, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)
	#	y_pred = tf.reduce_max(y_pred * skills, axis=-1, keepdims=True)
	#	return y_true, y_pred

	#def closs(y_true, y_pred):
	#	true, pred = get_target(y_true, y_pred)
	#	return binary_crossentropy(true, pred)
	#
	#class Auc(tf.keras.metrics.AUC):
	#	def update_state(self, y_true, y_pred, sample_weight=None):
	#		true, pred = get_target(y_true, y_pred)
	#		super(Auc, self).update_state(y_true=true, y_pred=pred, sample_weight=sample_weight)

	dkt = Dkt(args=args)

	dkt.compile()
	#dkt.compile(run_eagerly=True, loss=closs, optimizer='adam', metrics=[Auc(name='auc')])

	train = dkt.get_data(train_seqs)
	val = dkt.get_data(val_seqs)
	test = dkt.get_data(test_seqs)

	#with open('../data/sub_same_len_seqs.bf', mode='rb') as f:
	#	seqs = pickle.load(f)
	#dkt.get_right_or_error_predict_seqs(seqs)
	#dkt.last_acc(seqs)

	#for ds in train:
	#	preds = dkt(ds[0])
	#	print(preds)

	if args.test: # train and test 
		#t = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y_%m_%d_%H_%M_%S') #p = args.model_dir + t + '&' + str(args.epochs) + '_epochs&' + str(args.batch_size) + '_batch_size/'
		#if not os.path.exists(p):
		#        os.mkdir(p)
		#f = p + '{epoch:03d}epoch_{val_auc:.2f}val_auc_{val_loss:.2f}val_loss.bf'

		#mc = ModelCheckpoint(f, save_weights_only=True)
		#mc = ModelCheckpoint('../data/model/xxx/model{epoch:03d}epoch.h5')

		#dkt.fit(train, epochs=args.epochs, validation_data=val, callbacks=[mc])
		#dkt.fit(train, epochs=args.epochs, validation_data=val)
		dkt.fit(train, epochs=1)
		dkt.evaluate(test)
	else: # only test
		#dkt.fit(test.take(1))
		#dkt.load_weights('../data/model/xxx/checkpoint')
		#dkt.load_weights('../data/model/xxx.h5')
		#dkt = load_model('../data/model/xxx.h5')
		dkt.evaluate(test)



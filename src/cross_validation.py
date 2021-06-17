import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time, pickle, argparse, datetime, os, ast
import numpy as np
import tensorflow as tf
#from tensorflow.keras.models import load_model
#from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.metrics import binary_crossentropy

from dkt import Dkt
from akt_data.data import load_data

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Dkt model train and test')

	parser.add_argument("--train", default=False, action='store_true', help='train or test the model')
	parser.add_argument("--test", default=False, action='store_true', help='test the model')
	parser.add_argument("--cross_validation", default=False, action='store_true', help='k-fold cross validation')

	parser.add_argument('--gpus', type=str, default='0', help='want to using gpus, e.g "python main.py --gpus=0,1,2,3"')

	parser.add_argument('--epochs', type=int, default=300)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--lstm_dim', type=int, default=100)
	parser.add_argument('--dropout_rate', type=float, default=0.5)
	parser.add_argument('--mask_value', type=float, default=-1.0)
	parser.add_argument('--optimizer', type=str, default='adam')

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

	#concepts = [1223, 110, 100, 102, 188]
	#if dataset == 'statics':
	#	args.concepts = concepts[0]
	#if dataset == 'assist2009_pid':
	#	args.concepts = concepts[1]
	#if dataset == 'assist2015':
	#	args.concepts = concepts[2]
	#if dataset == 'assist2017_pid':
	#	args.concepts = concepts[3]

	concepts = {'statics': 1223, 'assist2009_pid': 110, 'assist2015': 100, 'assist2017_pid': 102, 'ednet': 188}
	args.concepts = concepts[dataset]
	print(args)
	
	# gpus setup
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
	print('\nCUDA_VISIBLE_DEVICES Environment Varaible:', os.environ['CUDA_VISIBLE_DEVICES'])
	gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
	for gpu in gpus:	# limit gpu memory
		tf.config.experimental.set_memory_growth(gpu, True)
	print('Used GPU Devices Number:', len(gpus), '\n')

	#dkt = Dkt(args=args)
	
	results = []
	if args.cross_validation:	# k-fold cross validation
		for i in range(5):
			print('\nFold', i)
			dkt = Dkt(args=args)
			#train_data_dir = '../data/akt_data/' + args.dataset + '/' + args.dataset + '_train' + str(i+1) + '.csv'
			#valid_data_dir = '../data/akt_data/' + args.dataset + '/' + args.dataset + '_valid' + str(i+1) + '.csv'
			#test_data_dir  = '../data/akt_data/' + args.dataset + '/' + args.dataset + '_test' + str(i+1) + '.csv'
			train_data_dir = '../data/akt_data/' + args.dataset + '/' + args.dataset + '_train' + str(i+1) + '.csv'
			valid_data_dir = '../data/akt_data/' + args.dataset + '/' + args.dataset + '_valid' + str(i+1) + '.csv'
			test_data_dir  = '../data/akt_data/' + args.dataset + '/' + args.dataset + '_test' + str(i+1) + '.csv'
			train_seqs	= load_data(train_data_dir, args.dataset)
			valid_seqs	= load_data(valid_data_dir, args.dataset)
			test_seqs	= load_data(test_data_dir, args.dataset)
			# train and test
			result = dkt.train_and_test(train_seqs, test_seqs, valid_seqs, args)
			results.append((result[1], result[2]))
		path = '../data/result/akt_data/' + args.dataset + '.txt'
		with open(path, mode='wt') as f:
			f.write('AUC:\n')
			for r in results:
				f.write(str(r[0]) + '\n')
			f.write('ACC:\n')
			for r in results:
				f.write(str(r[1]) + '\n')
			

	if args.train:	# train and test 
		train_data_dir = '../data/akt_data/' + args.dataset + '/' + args.dataset + '_train1.csv'
		valid_data_dir = '../data/akt_data/' + args.dataset + '/' + args.dataset + '_valid1.csv'
		test_data_dir  = '../data/akt_data/' + args.dataset + '/' + args.dataset + '_test1.csv'
		train_seqs	= load_data(train_data_dir, 4)
		valid_seqs	= load_data(valid_data_dir, 4)
		test_seqs	= load_data(test_data_dir, 4)

		dkt = Dkt(args=args)
		dkt.train_and_test(train_seqs, test_seqs, valid_seqs, args)
	if args.test:	# only test
		test_data_dir = '../data/akt_data/' + args.dataset + '/' + args.dataset + '_test1.csv'
		test_seqs	= load_data(test_data_dir, 4)
		dkt.evaluate(test_seqs)



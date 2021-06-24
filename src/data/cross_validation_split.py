import sys
sys.path.append(".")
import pickle
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from data.ednet_data import split_2_200

#with open('../data/ednet/seqs.bf', mode='br') as f:
#	seqs = pickle.load(f)	
#
#kf = KFold(n_splits=5)
#
#for index, (train, test) in enumerate(kf.split(seqs)):
#	train = np.array(seqs)[train]
#	test = np.array(seqs)[test]
#	train, val = train_test_split(train, test_size=0.25, shuffle=True)
#
#	with open('../data/ednet/train_' + str(index) + '.bf', mode='wb') as f:
#		pickle.dump(train, f)
#
#	with open('../data/ednet/val_' + str(index) + '.bf', mode='wb') as f:
#		pickle.dump(val, f)
#
#	with open('../data/ednet/test_' + str(index) + '.bf', mode='wb') as f:
#		pickle.dump(test, f)


for name in ['train', 'val', 'test']:
	for j in range(5):
		print(j)
		with open('../data/ednet/' + name + '_' + str(j) + '.bf', mode='rb') as f:
			seqs = pickle.load(f)

		new_seqs = split_2_200(seqs)
		with open('../data/ednet200/' + name + '_' + str(j) + '.bf', mode='wb') as f:
			pickle.dump(new_seqs, f)



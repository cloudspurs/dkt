from dkt import Dkt

import math
import pickle
import random
import numpy as np

### ednets dataset
#with open('../data/sub_seqs.bf', mode='rb') as f:
with open('../data/seqs.bf', mode='rb') as f:
	seqs = pickle.load(f)
num_skills = 188

#seqs = seqs[:784300]

#np.random.shuffle(seqs)
number = int(len(seqs)*0.8)
train = seqs[:number]
test = seqs[number:]

#print('\nSkills: ', num_skills)
print('All Seqs: ', len(seqs))
print('Train Seqs: ', len(train))
print('Test Seqs: ', len(test))

model = Dkt(num_skills)
model.ttrain_and_test(train, test, 20)

#model.train_and_test('../data/data.csv')
#model.train_and_test('../data/sub_data.csv')


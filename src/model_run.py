import numpy as np
from multi_dkt import MultiDkt
import time, math, pickle


# model parameters
epochs = 300
batch_size = 124
lstm_hiddent_unit = 100
dropout_rate = 0.5
optimizer = 'adam'

print('\nEpochs:', epochs, '\nBatch Size:', batch_size, '\nlstm_hidden_units:', lstm_hiddent_unit, '\ndropout_rate:', dropout_rate, '\noptimizer:', optimizer)

## read ednets dataset
s = time.time()
#np.random.shuffle(seqs)

# ednet (multi concept as one new concept)
#with open('../data/one_concept/sub_seqs.bf', mode='rb') as f:
#with open('../data/one_concept/seqs.bf', mode='rb') as f:
#	seqs = pickle.load(f)
#	num_skills = 1791

# ednet
num_skills = 188
with open('../data/ednet/sub_seqs.bf', mode='rb') as f:
#with open('../data/seqs.bf', mode='rb') as f:
#with open('../data/ednet200-1/split_200_morethan_10.bf', mode='rb') as f:
	seqs = pickle.load(f)

# split train, test, valid
length = len(seqs)
a = int(length*0.8)
b = int(a*0.8)
train_seqs = seqs[:b]
val_seqs = seqs[b:a]
test_seqs = seqs[a:]

# statistics infos
#print('students num:', len(seqs))
#questions = 0
#for seq in seqs:
#	questions += len(seq)
#print('questions num:', questions)
#print('Train + Val + Test:', len(train_seqs) + len(val_seqs) + len(test_seqs))
#

#with open('../data/ednet/train_seqs.bf', mode='rb') as f:
#	train_seqs = pickle.load(f)
#with open('../data/ednet/val_seqs.bf', mode='rb') as f:
#	val_seqs = pickle.load(f)
#with open('../data/ednet/test_seqs.bf', mode='rb') as f:
#	test_seqs = pickle.load(f)

#num_skills = 110
#with open('../data/assist2009/train_seqs.bf', mode='rb') as f:
#	train_seqs = pickle.load(f)
#with open('../data/assist2009/val_seqs.bf', mode='rb') as f:
#	val_seqs = pickle.load(f)
#with open('../data/assist2009/test_seqs.bf', mode='rb') as f:
#	test_seqs = pickle.load(f)

e = time.time()
print('\nRead seqs time:', int(e-s), 's')
print('\nSkills:', num_skills)

# model define and train 
model = MultiDkt(num_skills, batch_size=batch_size, hidden_units=lstm_hiddent_unit, optimizer=optimizer, dropout_rate=dropout_rate)
model.train_and_test(train_seqs, val_seqs, test_seqs, epochs=epochs)


#model.load_weights()
#model.get_right_or_error_predict_seqs(seqs)
#model.last_acc(seqs)
#model.predict(seqs)
#model.evaluate(seqs)

#model.load_weights('../data/model/ednet_200_no_delete_morethan_10/dkt.h5')
#model.evaluate(test_seqs)
#model.save_lrp_weights()


import pickle
import numpy as np

with open('../data/model/lrp_ednet_4.weights', mode='rb') as f:
	weights = pickle.load(f) 

w_xh_1 = weights['w_xh']
w_hh_1 = weights['w_hh']
b_xh_h_1 = weights['b_xh_h']
w_hy_1 = weights['w_hy']
b_hy_1 = weights['b_hy']

with open('../data/model/lrp_ednet_0.weights', mode='rb') as f:
#with open('../data/model/lstm_weights', mode='rb') as f:
	weights = pickle.load(f) 

w_xh_2 = weights['w_xh']
w_hh_2 = weights['w_hh']
b_xh_h_2 = weights['b_xh_h']
w_hy_2 = weights['w_hy']
b_hy_2 = weights['b_hy']

print('w_xh:', np.mean(w_xh_1-w_xh_2))
print('w_hh:', np.mean(w_hh_1-w_hh_2))
print('b_xh_h:', np.mean(b_xh_h_1-b_xh_h_2))
print('w_hy:', np.mean(w_hy_1-w_hy_2))
print('b_hy:', np.mean(b_hy_1-b_hy_2))


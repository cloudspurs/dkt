'''
@author: Leila Arras
@maintainer: Leila Arras
@date: 21.06.2017
@version: 1.0+
@copyright: Copyright (c) 2017, Leila Arras, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license: see LICENSE file in repository root
'''

#import sys
#sys.path.append('../')

import pickle
import numpy as np
from numpy import newaxis as na
from lrp_linear_layer import *


class lstm:
	def __init__(self, model_path='../../data/model/dkt_lrp.weights'):
	#def __init__(self, model_path='../model/lstm_weights'):

		# load model weights
		with open(model_path, mode='rb') as f:
			model = pickle.load(f)

		# LSTM left encoder
		self.w_xh= model["w_xh"]  # shape (4*50, 376)
		self.w_hh= model["w_hh"]  # shape (4*50, 50)
		self.b_xh_h = model["b_xh_h"]  # shape (4*50,)

		# linear output layer
		self.w_hy= model["w_hy"] # shape (188, 50)
		self.b_hy = model['b_hy'] # shape (188,)

		self.f_w = model['f_w'] # (32, 376)
		self.f_b = model['f_b'] # (32,)
		self.z_w = model['z_w'] # (1, 32)
		self.z_b = model['z_b'] # (1,)

		self.num_skills = self.w_hy.shape[0] # 188
	

	def set_input(self, w, delete_pos=None):
		T = len(w)							# sequence length
		d = int(self.w_xh.shape[0]/4)		# 50 hidden layer dimension
		e = self.w_xh.shape[1]				# 376 input embedding dimension
		x = np.zeros((T, e))
		qtts = np.zeros((T, int(e/2)))
		
		#for t in range(T):
		#	skill, is_correct = w[t]
		#	feature, qtt, label = one_hot(skill, is_correct, self.w_hy.shape[0])
		#	x[t] = feature
		#	qtts[t] = qtt

 		#x[0] = np.zeros((e))
 		#for t in range(T-1):
 		#    skill, is_correct = w[t]
 		#    feature, label = one_hot(skill, is_correct, self.w_hy.shape[0])
 		#    x[t+1] = feature

		for t in range(T):
			skills = w[t][0]
			answer = w[t][1]
			xt = np.zeros((2 * self.num_skills,))
			qtt = np.zeros((self.num_skills,))
			for skill in skills:
				xt[answer * self.num_skills + skill] = 1
				qtt[skill] = 1
			x[t] = xt
			qtts[t] = qtt

		if delete_pos is not None:
			x[delete_pos, :] = np.zeros((len(delete_pos), e))
		
		self.w = w
		self.x = x[:-1]
		self.qtts = qtts[1:]
		self.h = np.zeros((T, d))
		self.c = np.zeros((T, d))
	
   
	def forward(self):
		"""
		Standard forward pass.
		Compute the hidden layer values (assuming input x/x_rev was previously set)
		"""
		T = len(self.w) - 1							
		d = int(self.w_xh.shape[0]/4) 
		# gate indices (assuming the gate ordering in the LSTM weights is i,g,f,o):		
		idx = np.hstack((np.arange(0,d), np.arange(2*d,4*d))).astype(int) # indices of gates i,f,o together
		idx_i, idx_g, idx_f, idx_o = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d) # indices of gates i,g,f,o separately
		  
		# initialize
		self.gates_xh  = np.zeros((T, 4*d))  
		self.gates_hh  = np.zeros((T, 4*d)) 
		self.gates_pre = np.zeros((T, 4*d))  # gates pre-activation
		self.gates	   = np.zeros((T, 4*d))  # gates activation
			 
		for t in range(T): 
			self.gates_xh[t]	 = np.dot(self.w_xh, self.x[t]) 
			self.gates_hh[t]	 = np.dot(self.w_hh, self.h[t-1]) 
			self.gates_pre[t]	 = self.gates_xh[t] + self.gates_hh[t] + self.b_xh_h
			#self.gates_pre[t]	 = self.gates_xh[t] + self.gates_hh[t]
			self.gates[t,idx]	 = 1.0/(1.0 + np.exp(-self.gates_pre[t,idx]))
			self.gates[t,idx_g]  = np.tanh(self.gates_pre[t,idx_g]) 
			self.c[t]			 = self.gates[t,idx_f]*self.c[t-1] + self.gates[t,idx_i]*self.gates[t,idx_g]
			self.h[t]			 = self.gates[t,idx_o]*np.tanh(self.c[t])

			
		### only need the last time y, f, z
		#self.y = np.dot(self.w_hy,	self.h[-1])
		self.y = np.dot(self.w_hy,  self.h[T-1]) + self.b_hy
		self.s = 1.0 / (1.0 + np.exp(-self.y))

		self.con = np.concatenate((self.s, self.qtts[-1]))
		self.f = np.dot(self.f_w, self.con) + self.f_b
		self.f_a = 1.0 / (1.0 + np.exp(-self.f))
		self.z = np.dot(self.z_w, self.f_a) + self.z_b
		self.z_a = 1.0 / (1.0 + np.exp(-self.z))
		print(self.z_a, 'The last result')
		
		return self.s.copy() # prediction scores
	
				   
	def lrp(self, w, LRP_class, eps=0.001, bias_factor=1.0):
		"""
		Layer-wise Relevance Propagation (LRP) backward pass.
		Compute the hidden layer relevances by performing LRP for the target class LRP_class
		(according to the papers:
			- https://doi.org/10.1371/journal.pone.0130140
			- https://doi.org/10.18653/v1/W17-5221 )
		"""
		# forward pass
		self.set_input(w)
		self.forward() 
		
		T	   = len(self.w) - 1
		d	   = int(self.w_xh.shape[0]/4)
		e	   = self.w_xh.shape[1] 
		C	   = self.w_hy.shape[0]  # number of classes
		idx    = np.hstack((np.arange(0,d), np.arange(2*d,4*d))).astype(int) # indices of gates i,f,o together
		idx_i, idx_g, idx_f, idx_o = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d) # indices of gates i,g,f,o separately
		
		# initialize
		Rx = np.zeros(self.x.shape) # (7, 376)
		Rh	= np.zeros((T+1, d))
		Rc	= np.zeros((T+1, d))
		Rg	= np.zeros((T,	 d)) # gate g only

		Rf = lrp_linear(self.f_a, self.z_w.T, self.z_b, self.z, self.z_a, self.z_w.shape[1], eps, bias_factor)
		print(np.sum(Rf), 'Rf')

		Rcon = lrp_linear(self.con, self.f_w.T, self.f_b, self.f, Rf, self.f_w.shape[1], eps, bias_factor) 
		print(np.sum(Rcon), 'Rcon')
		print(np.sum(Rcon[:188]), np.sum(Rcon[188:]), 'Rxt, Rqtt')

		Rh[T-1] = lrp_linear(self.h[T-1], self.w_hy.T, self.b_hy, self.y, Rcon[:self.num_skills], d, eps, bias_factor) 
		print(np.sum(Rh[T-1]), 'Rh[T-1]')
		
		#Rout_mask			 = np.zeros((C))
		#Rout_mask[LRP_class] = 1.0	
		## format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
		#Rh[T-1]  = lrp_linear(self.h[T-1],	self.w_hy.T , np.zeros((C)), self.s, self.s*Rout_mask, 2*d, eps, bias_factor, debug=False)
		
		for t in reversed(range(T)):
			Rc[t]	+= Rh[t] # h ->c

			# c -> c(t-1), c -> g
			Rc[t-1]  = lrp_linear(self.gates[t,idx_f]*self.c[t-1], np.identity(d), np.zeros((d)), self.c[t], Rc[t], 2*d, eps, bias_factor, debug=False)
			Rg[t]	 = lrp_linear(self.gates[t,idx_i]*self.gates[t,idx_g], np.identity(d), np.zeros((d)), self.c[t], Rc[t], 2*d, eps, bias_factor, debug=False)

			# g -> x, g -> h
			Rx[t]	 = lrp_linear(self.x[t], self.w_xh[idx_g].T, self.b_xh_h[idx_g], self.gates_pre[t,idx_g], Rg[t], d+e, eps, bias_factor, debug=False)
			Rh[t-1]  = lrp_linear(self.h[t-1], self.w_hh[idx_g].T, self.b_xh_h[idx_g], self.gates_pre[t,idx_g], Rg[t], d+e, eps, bias_factor, debug=False)
			
		print(Rx.sum(), 'Rx')
		print(Rh[-1].sum(), 'Rh[-1]')
		print(Rc[-1].sum(), 'Rc[-1]')
		print(Rx.sum() + Rh[-1].sum() + Rc[-1].sum(), 'Rx + Rh + Rc')
		return Rx, Rh[-1].sum()+Rc[-1].sum()


# 将edent数据转换成学生作答序列格式
# 原来每行代表一个作答，现每两行代表一个学生的全部作答序列，
# 第一行题目id，第二行作答对错

import time, pickle
import pandas as pd

def seqs(f):
	df = pd.read_csv(f)	
	a = 0
	for index, row in df.iterrows():
		#if index % 1000 == 0:
		#	print(index)
		a += index
	print(a)

	#df = df.groupby('stu_docu_id')
		

def seqs2(p, out):
	with open(p, mode='rb') as f:
		seqs = pickle.load(f)
		#print(seqs[0])
	
	with open(out, mode='wt') as f:
		for seq in seqs:
			#for item in seq:
			#	t = list(map(str, item[0]))
			#	a = '_'.join(t)
			#	ids += a + ','
			#	correct_wrong += str(item[1]) + ','

			ids = ''
			t = [list(map(str, e[0])) for e in seq]
			t = ['_'.join(e) for e in t]
			ids += ','.join(t)
			
			t = [str(e[1]) for e in seq]
			correct_wrong = ''
			correct_wrong += ','.join(t)

			#print(ids)
			#print(correct_wrong)
			f.write(ids + '\n')
			f.write(correct_wrong + '\n')

def bf_to_csv():
	for d in ['train', 'val', 'test']:
		print(d)
		for i in range(5):
			print(i)
			a = '../data/ednet200/' + d + '_' + str(i) + '.bf'
			b = '../data/ednet200/' + d + '_' + str(i) + '.csv'
			seqs2(a, b)

if __name__ == '__main__':
	#s = time.time()	
	#seqs('../data/ednet/data.csv')
	#e = time.time()
	#print('time', e-s)

	#seqs2('../data/ednet/seqs.bf', '../data/ednet/seqs.csv')
	bf_to_csv()

	#s = time.time()	
	#with open('../data/ednet/data.csv', mode='rt') as f:
	#	l = 0
	#	for i, line in enumerate(f):
	#		l += 1
	#	print(l)
	#e = time.time()
	#print('time', e-s)


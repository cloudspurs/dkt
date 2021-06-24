import pickle, time
import numpy as np
import pandas as pd


#with open('../data/seqs.bf', mode='rb') as f:
##with open('../data/sub_seqs.bf', mode='rb') as f:
#	seqs = pickle.load(f)
#	
#new_seqs = []
#for seq in seqs:
#	length = len(seq)
#	if length > 200:
#		for i in range(0, length, 200):
#			s = i
#			e = i+200
#			if e > length:
#				e = length
#			new_seqs.append(seq[s:e])
#	else:
#		new_seqs.append(seq)
#
#with open('../data/split_200_morethan_10_no_delete.bf', mode='wb') as f:
#	pickle.dump(new_seqs, f)	
#exit()
	


def split_2_200(seqs):
	result = []
	size = 200
	for seq in seqs:
		length = len(seq)
		#print('\n', length)
		#temp = [seqs[i:i+size] for i in range(0, length, size)]
		temp = []
		for i in range(0, length, size):
			s = i
			e = i + size
			if e > length:
				e = length
			temp.append(seq[s:e])

		#print(len(temp[-1]))
		result.extend(temp)	
		#if len(temp[-1]) >= 10:
		#	result.extend(temp)	
		#else:
		#	if len(temp) >= 2:
		#		result.extend(temp[:-1])
	return result


#with open('../data/ednet200-2/sub_seqs.bf', mode='rb') as f:
#with open('../data/ednet200-2/seqs.bf', mode='rb') as f:
#	seqs = pickle.load(f)
#
#length = len(seqs)
#for i in range(5):
#	start = int(length * 0.2 * i)
#	end = int(length * 0.2 * (i+1))
#
#	test = seqs[start:end] 
#	train = seqs[:start] + seqs[end:]
#
#	np.random.shuffle(train)
#	split = int(len(train) * 0.8)
#	val = train[split:]
#	train = train[:split]
#
#	with open('../data/ednet200-2/train_' + str(i) + '.bf', mode='wb') as f:
#		pickle.dump(train, f)
#	with open('../data/ednet200-2/val_' + str(i) + '.bf', mode='wb') as f:
#		pickle.dump(val, f)
#	with open('../data/ednet200-2/test_' + str(i) + '.bf', mode='wb') as f:
#		pickle.dump(test, f)
#
#	#train = split_2_200(train)
#	#val = split_2_200(val)
#	#test = split_2_200(test)
#
#	with open('../data/ednet200-2/200_train_' + str(i) + '.bf', mode='wb') as f:
#		pickle.dump(train, f)
#	with open('../data/ednet200-2/200_val_' + str(i) + '.bf', mode='wb') as f:
#		pickle.dump(val, f)
#	with open('../data/ednet200-2/200_test_' + str(i) + '.bf', mode='wb') as f:
#		pickle.dump(test, f)


s = time.time()
d_path = '../data/ednet/data.csv'
q_path = '../data/ednet/question.csv'

### 1. get question skills
skills = set()
question_2_skills = {}
df = pd.read_csv(q_path)
for i,r in df.iterrows():
	q = r['question_id']
	tags = r['tags'].split(';')
	s = [int(t) for t in tags]
	skills.update(s)
	question_2_skills[q] = s

skills = sorted(skills)

print(len(skills), 'Skill Number')
print(len(question_2_skills), 'Question Number')
#print(sorted(skills))
#print(question_2_skills)


### 2. process student answer data
#seqs = []
#df = pd.read_csv(d_path)
#num = 1
#for i,r in df.iterrows():
#	num = num + 1
#print(num)

students = {}
seqs = {}
num = 0
with open(d_path, mode='rt') as f:
	lines = f.readlines()
	for line in lines[1:]:
		num = num + 1
		items = line.split(',')

		f = items[-1] # 每个学生作答数据文件名，用来生成学生id
		qid  = items[2]
		answer = int(items[-2])

		# get student id
		if f in students.keys():
			sid = students[f]
		else:
			sid = len(students)
			students[f] = sid
		
		skills = question_2_skills[qid]
		# 把skill转换成从0开始的连续值
		new_skills = []
		for s in skills:
			new_skills.append(skills.index(s))

		if sid in seqs.keys():
			seqs[sid].append((new_skills, answer))
		else:
			seq = []
			seq.append((new_skills, answer))
			seqs[sid] = seq

seqs = [seqs[k] for k in seqs.keys()]
e = time.time()
print('time:', e-s)

#
#print(num, 'Question Number')
#print(len(seqs), 'Student Number')
#
#with open('../data/seqs.bf', mode='wb') as f:
#	pickle.dump(seqs, f)


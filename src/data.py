import pickle
import numpy as np
import pandas as pd


d_path = '../data/data.csv'
q_path = '../data/question.csv'

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

xxx = []
for k,v in question_2_skills.items():
	xxx.append(len(v))
print(np.max(xxx), 'max skills one question')

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

print(seqs[4])

seq_len = [len(seqs[k]) for k in seqs.keys()]
seq_len.sort()
print(seq_len[:100])
print('min 100 seqs len')
seq_len.reverse()
print(seq_len[:100])
print('max 100 seqs len')

max_len = np.max(seq_len)
print(max_len, 'Seq max length:')
max_len_stu = list(seqs.keys())[np.argmax(seq_len)]
for k,v in students.items():
	if v == max_len_stu:
		max_len_stu_name = k
print(max_len_stu_name, 'Seq max length stu:')
exit()

seqs = [seqs[k] for k in seqs.keys()]

print(num, 'Answer Number')
print(len(seqs), 'Student Number')

with open('../data/seqs.bf', mode='wb') as f:
	pickle.dump(seqs, f)


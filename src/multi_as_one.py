import pickle
import numpy as np
import pandas as pd


d_path = '../data/data.csv'
q_path = '../data/question.csv'

### 1. get question skills
skills = set() # 存放单个知识点
multi_skill_2_sorted_id = {} # 每个题目的多知识点生成新的单知识点
question_2_skills = {} # 每个题目对应的多知识点id
question_2_one_id = {} # 每个题目对应的合并后的知识点id

df = pd.read_csv(q_path)
for i,r in df.iterrows():
	q = r['question_id']
	tags = r['tags'].split(';')

	s = [int(t) for t in tags]
	skills.update(s)
	question_2_skills[q] = s

	#x = set(s)
	#if len(s) != len(x):
	#	print(s)
	#	print(x)

	ms = '_'.join(tags)
	if ms not in multi_skill_2_sorted_id.keys():
		multi_skill_2_sorted_id[ms] = len(multi_skill_2_sorted_id)
		question_2_one_id[q] = multi_skill_2_sorted_id[ms]
	else:
		question_2_one_id[q] = multi_skill_2_sorted_id[ms]

skills = sorted(skills)
print(len(skills), 'Skill Number')
print(len(multi_skill_2_sorted_id), 'One Skill Nuber')
print(len(question_2_skills), 'Question Number')
exit()


### 2. process student answer data
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
		
		skill = question_2_one_id[qid]

		if sid in seqs.keys():
			seqs[sid].append((skill, answer))
		else:
			seq = []
			seq.append((skill, answer))
			seqs[sid] = seq

seqs = [seqs[k] for k in seqs.keys()]

print(num, 'Answer Number')
print(len(seqs), 'Student Number')

same_len_seqs = []
for seq in seqs:
	small_seqs = [seq[i:i+200] for i in range(0, len(seq), 200)]
	same_len_seqs.extend(small_seqs)

with open('../data/one_concept/seqs.bf', mode='wb') as f:
	pickle.dump(same_len_seqs, f)


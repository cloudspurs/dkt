### combine question and answer files
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

skills = sorted(skills)

print(len(skills), 'Skill Number')
print(len(question_2_skills), 'Question Number')


### 2. process student answer data
#seqs = []
#df = pd.read_csv(d_path)
#num = 1
#for i,r in df.iterrows():
#	num = num + 1
#print(num)

with open(d_path, mode='rt') as f:
	lines = f.readlines()

	with open('../data/new_data.csv', mode='rt') as nf:
		for line in lines[1:]:
			items = line.split(',')

			qid  = items[2]

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
			line = 


with open('../data/seqs.bf', mode='wb') as f:
	pickle.dump(seqs, f)


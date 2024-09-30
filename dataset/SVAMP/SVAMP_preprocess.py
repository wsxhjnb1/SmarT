import csv
import json
file_path = 'train.csv'

number = []
question = []
with open(file_path, mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for i, row in enumerate(csv_reader):
        if i == 0:
            continue
        question.append(row[0].replace(" . ", ". ").replace(" , ", ", ").strip())
        number.append(row[1].strip())

corpus = []
for n,q in zip(number, question):
    n = n.split()
    for i,value in enumerate(n):
        number_str = "number{}".format(i)
        q = q.replace(number_str,value)
    corpus.append(q)

with open('SVAMP_corpus.json', 'w') as f:
    json.dump(corpus, f)


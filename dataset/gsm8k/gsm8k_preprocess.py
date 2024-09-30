import json
with open('train.jsonl', 'r') as f:
    train = f.read()
train_strs = train.strip().split('\n')

train = []
for train_str in train_strs:
    obj = json.loads(train_str)
    train.append(obj)


corpus = []
for item in train:
    question = item['question']
    corpus.append(question)

with open('gsm8k_corpus.json', 'w') as f:
    json.dump(corpus,f)

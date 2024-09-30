import json
with open('train_rand_split.jsonl', 'r') as f:
    train = f.read()
train_strs = train.strip().split('\n')

train = []
for train_str in train_strs:
    obj = json.loads(train_str)
    train.append(obj)


corpus = []
for item in train:
    choice = "Answer Choices:"
    for c in item["question"]["choices"]:
        choice += " ("
        choice += c["label"]
        choice += ") "
        choice += c["text"]
    q = item["question"]["stem"].strip() + " " + choice
    corpus.append(q)

with open('CommonsenseQA_corpus.json', 'w') as f:
    json.dump(corpus,f)

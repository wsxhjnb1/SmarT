import json
with open("strategyqa_test.json",'r') as f:
    train = json.load(f)

corpus = []
for item in train:
    corpus.append(item["question"])

with open("StrategyQA_corpus.json", 'w') as f:
    json.dump(corpus,f)
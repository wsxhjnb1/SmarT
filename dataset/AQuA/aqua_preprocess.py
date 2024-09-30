import json

with open('train.json', 'r') as f:
    train = f.read()

train_strs = train.strip().split('\n')

train = []
for train_str in train_strs:
    obj = json.loads(train_str)
    train.append(obj)

corpus = []
sum = 0
sum2 = 0
op = ["A", "B", "C", "D", "E"]
for item in train:
    question = item['question']
    for i in range(len(item['options'])):
        # if '{})({})'.format(op[i], op[i].lower()) in item['options'][i]:
        item['options'][i] = item['options'][i].replace('{})({}) '.format(op[i], op[i].lower()), '{})'.format(op[i]))
        item['options'][i] = item['options'][i].replace('{})({}) '.format(op[i].lower(), op[i]), '{})'.format(op[i]))
        item['options'][i] = item['options'][i].replace('{})({}) '.format(op[i], op[i]), '{})'.format(op[i]))
        item['options'][i] = item['options'][i].replace('{})({}) '.format(op[i].lower(), op[i].lower()), '{})'.format(op[i]))
        item['options'][i] = item['options'][i].replace('{})({})'.format(op[i].lower(), op[i]), '{})'.format(op[i]))
        item['options'][i] = item['options'][i].replace('{})({})'.format(op[i], op[i]), '{})'.format(op[i]))
        item['options'][i] = item['options'][i].replace('{})({})'.format(op[i].lower(), op[i].lower()), '{})'.format(op[i]))
        item['options'][i] = item['options'][i].replace('{})({})'.format(op[i], op[i].lower()), '{})'.format(op[i]))

        item['options'][i] = item['options'][i].replace('{}){}) '.format(op[i], op[i].lower()), '{})'.format(op[i]))
        item['options'][i] = item['options'][i].replace('{}){}) '.format(op[i].lower(), op[i]), '{})'.format(op[i]))
        item['options'][i] = item['options'][i].replace('{}){}) '.format(op[i], op[i]), '{})'.format(op[i]))
        item['options'][i] = item['options'][i].replace('{}){}) '.format(op[i].lower(), op[i].lower()), '{})'.format(op[i]))
        item['options'][i] = item['options'][i].replace('{}){})'.format(op[i].lower(), op[i]), '{})'.format(op[i]))
        item['options'][i] = item['options'][i].replace('{}){})'.format(op[i], op[i]), '{})'.format(op[i]))
        item['options'][i] = item['options'][i].replace('{}){})'.format(op[i].lower(), op[i].lower()), '{})'.format(op[i]))
        item['options'][i] = item['options'][i].replace('{}){})'.format(op[i], op[i].lower()), '{})'.format(op[i]))

        item['options'][i] = item['options'][i].replace('{}) '.format(op[i].lower()), '{})'.format(op[i]))
        item['options'][i] = item['options'][i].replace('{}) '.format(op[i]), '{})'.format(op[i]))
    choice = "(" + "(".join(item['options'])
    choice = choice.replace("(", " (").replace(")", ") ")
    choice = "Answer Choices:" + choice.replace('(A)  ', '(A) ').replace('(B)  ', '(B) ')\
        .replace('(C)  ', '(C) ').replace('(D)  ', '(D) ').replace('(E)  ', '(E) ')
    if "cannot be determined" in choice.lower() or "none" in choice.lower():
        continue
    sum2 += 1
    if "(A)  " in choice:
        sum += 1
    corpus.append(question + '\n' + choice)

with open('AQuA_corpus.json', 'w') as f:
    json.dump(corpus,f)
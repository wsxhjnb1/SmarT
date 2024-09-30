import pandas as pd
import json
import random

def generate_sample_for_last_letters():
    male_first_name = pd.read_excel('First_name1000-2000.xlsx', sheet_name='Sheet1', skiprows=1, usecols="B")
    female_first_name = pd.read_excel('First_name1000-2000.xlsx', sheet_name='Sheet1', skiprows=1, usecols="E")
    first_name = []
    for item in male_first_name.values:
        first_name.extend(item)
    for item in female_first_name.values:
        first_name.extend(item)

    all_last_name = pd.read_excel('Last_name_1000-2000.xlsx', sheet_name='Sheet1', skiprows=0, usecols="B")
    last_name = []
    for item in all_last_name.values:
        last_name.extend(item)

    Template1 = "Take the last letters of each words in \""
    Template2 = "\" and concatenate them."

    first_name_sample = random.sample(first_name,100)
    last_name_sample = random.sample(last_name,100)
    samples = []
    for i in range(100):
        samples.append(Template1 + first_name_sample[i].title() +' '+ last_name_sample[i].title() +Template2)
    print(samples)
    with open('./last_letters_corpus.json','w') as f:
        json.dump(samples,f)

if __name__ == '__main__':
    generate_sample_for_last_letters()
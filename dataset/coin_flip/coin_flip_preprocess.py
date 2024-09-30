import json
import openai
import os

openai.api_key = ""

def create_response_chat(prompt_input, n, eng, max_tokens=256, temperature=0.0, stop="Q", top_p=1):
    response = openai.ChatCompletion.create(
        model=eng,
        messages=prompt_input,
        temperature=temperature,
        n=n,
        stop=stop,
        max_tokens = max_tokens,
        top_p=top_p
    )
    return response

def decoder_for_gpt3(input, temperature, max_length, n = 1, top_p = 1, stop = "A:"):
    response = create_response_chat([
        {"role": "system", "content": "Follow the given examples and answer the question."},
        {"role": "user", "content": input},
    ], n, "gpt-3.5-turbo-0613", max_length, temperature, stop=stop,top_p=top_p)
    if n > 1:
        answers = []
        for i in range(n):
            answer = response['choices'][i]['message']["content"].strip()
            answers.append(answer)
        return answers
    else:
        return response['choices'][0]['message']["content"].strip()

with open(os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/demos/coin_flip.txt', "r", encoding='utf-8') as f:
    prompt = f.read().strip()

inputs = prompt + '\nQ:'

pred = decoder_for_gpt3(inputs, 1, 256, n=128)
pred = list(set(pred))
preds = []
for item in pred:
    all_action = item.count("flips the coin")
    neg_action = item.count("not flips the coin")
    pos_action = all_action - neg_action
    if pos_action != 2 and item != "":
        preds.append(item)

with open("coin_flip_corpus.json", "w") as f:
    json.dump(preds,f)
import openai
import time
import asyncio

def create_response(prompt_input, eng='text-davinci-003', max_tokens=256, temperature=0.0, stop="Q"):
    response = openai.Completion.create(
        engine=eng,
        prompt=prompt_input,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["{}:".format(stop)]
    )
    return response

def create_response_chat(prompt_input, n, eng='gpt-3.5-turbo', max_tokens=256, temperature=0.0, stop="Q", top_p=1):
    response = openai.ChatCompletion.create(
        model=eng,
        messages=prompt_input,
        temperature=temperature,
        n=n,
        stop=stop,
        top_p=top_p
    )
    return response

def decoder_for_gpt3(args, input, temperature, max_length, apikey, n = 1, top_p = 1, stop = "Q:"):
    openai.api_key = apikey
    response = create_response_chat([
        {"role": "system", "content": "Follow the given examples and answer the question."},
        {"role": "user", "content": input},
    ], n, args.engine, max_length, temperature, stop=stop,top_p=top_p)
    if n > 1:
        answers = []
        for i in range(n):
            answer = response['choices'][i]['message']["content"].strip()
            answers.append(answer)
        return answers
    else:
        return response['choices'][0]['message']["content"].strip()



def basic_runner(args, inputs, temperature, max_length, apikey, max_retry=3, n=1, top_p=1, stop = "Q:"):
    retry = 0
    get_result = False
    pred = ''
    error_msg = ''
    sleep_time = args.api_time_interval
    while not get_result:
        try:
            pred = decoder_for_gpt3(args, inputs, temperature, max_length, apikey, n,top_p=top_p, stop= stop)
            get_result = True
        except Exception as e:
            if retry < max_retry:
                    time.sleep(sleep_time)
                    retry += 1
                    sleep_time = sleep_time * (retry + 1)
            else:
                error_msg = e.user_message
                break
    return get_result, pred, error_msg

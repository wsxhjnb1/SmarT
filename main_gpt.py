import json
from utils import load_data,print_now,mkpath
from config import args
from prompt import get_prompt
from gpt_runner import basic_runner
from utils import write_json, load_unlabel_data
import re
import copy
import logging
import os
import math
import random
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


now = print_now(1).split(' ')[0].replace('/', '-')
now2 = print_now(1).split(' ')[1].replace('/', '-')
Result_Folder = 'results/{}'.format(now)
mkpath('results')
mkpath(Result_Folder)
mkpath(f'{Result_Folder}/{args.dataset}')

Log_Folder = 'log/{}'.format(now)
mkpath('log')
mkpath(Log_Folder)
mkpath(f'{Log_Folder}/{args.dataset}')


Decoder_Error_File = f'{Result_Folder}/{args.dataset}-{args.engine}--Mode:{args.mode}--{now2}_deco.json'
Predict_File = f'{Result_Folder}/{args.dataset}/{args.engine}--question_set:{args.sample_question_num}--sample_num:{args.sample_num}--{now2}.json'
Log_File = f'{Log_Folder}/{args.dataset}/{args.engine}--Mode:{args.mode}--{now2}.log'


logging.basicConfig(filename=Log_File,level=logging.WARNING,filemode='a')
formatter = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

sh = logging.StreamHandler()
sh.setFormatter(formatter)
sh.setLevel(logging.WARNING)

fh = logging.FileHandler(filename=Log_File)
fh.setFormatter(formatter)
fh.setLevel(logging.WARNING)

logger = logging.getLogger()
logger.addHandler(fh)
logger.addHandler(sh)


def normalized_entropy(dict):
    key_num = len(dict.keys())
    if key_num == 1:
        return 1
    value_sum = 0
    aaa = 0
    for value in dict.values():
        value_sum += value
    for value in dict.values():
        probality = value/value_sum
        aaa -= math.log(probality,2)*probality
    return 1 - aaa/math.log(key_num,2)

def nsmall1(dataset, n,list,max,fsc_cot):
    nums=copy.deepcopy(list)
    temp = []
    Inf = max
    i = 0
    if dataset == 'coin_flip':
        half_sample_num = args.sample_num/2
        yes_num = 0
        no_num = 0
        while i < n:
            index = nums.index(min(nums))
            if fsc_cot[index] != None:
                if "yes" in fsc_cot[index] and yes_num < half_sample_num:
                    nums[index] = Inf
                    temp.append(index)
                    yes_num += 1
                    i += 1
                    continue
                if "no" in fsc_cot[index] and no_num < (args.sample_num - half_sample_num):
                    nums[index] = Inf
                    temp.append(index)
                    no_num += 1
                    i += 1
                    continue
                nums[index] = Inf
            else:
                nums[index] = Inf
        random.shuffle(temp)
    else:
        while i < n:
            index = nums.index(min(nums))
            if fsc_cot[index] != None:
                nums[index] = Inf
                temp.append(index)
                i += 1
            else:
                nums[index] = Inf
    return temp


def text_cleanup(text, stopword):
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    lemmatizer = WordNetLemmatizer()
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopword.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [token.strip() for token in tokens]
    tokens = [token for token in tokens if not token.isdigit()]
    return tokens

def get_precision(gt_ans: float) -> int:
    precision = 5
    if '.' in str(gt_ans):
        precision = len(str(gt_ans).split('.')[-1])
    return precision

def extract_number(args, text: str):
    text = text.replace(',', '')
    if '/' in text:
        pred = text.split()
        for token in pred:
            if '/' in token:
                token = re.sub(r'[^0-9\/]', '', token)
                return token
        return None
    else:
        pred = [s for s in re.findall(r'-?\d+\.?\d*', text)]
        if pred:
            pred_answer = float(pred[-1])
        else:
            pred_answer = None
    return pred_answer

def extract_answer(args, text):
    dataset = args.dataset.lower()
    if dataset in ["svamp", "gsm8k", "multiarith", "addsub", "singleeq"]:
        pred_answer = extract_number(args, text)
    elif dataset == "commonsenseqa":
        pred = text.strip()
        pred = re.sub("\(|\)|\:|\.|\,", "", pred)
        pred = pred.split()
        pred_answer = [i for i in pred if i in ('A|B|C|D|E')][-1]
        # pred_answer = re.findall(r'A|B|C|D|E', pred)[0]
        return pred_answer
    elif dataset == "aqua":
        pred = text.strip()
        pred_answer = re.findall(r'A|B|C|D|E', pred)[0]
        return pred_answer
    elif dataset == "strategyqa":
        if "unknown" in text.lower() or "uncertain" in text.lower():
            return "no"
        pred = text.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split()
        pred_answer = [i for i in pred if i in ("yes", "no")][-1]
        return pred_answer
    elif dataset == 'coin_flip':
        pred = text.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split()
        pred_answer = [i for i in pred if i in ("yes", "no")][-1]
        return pred_answer
    elif dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s", "", text)
        pred_answer = pred
        return pred_answer
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))

    if isinstance(pred_answer, str):
        if "/" in pred_answer:
            Numerator, Denominator = pred_answer.split('/')
            if float(Denominator) != 0:
                pred_answer = float(Numerator) / float(Denominator)
            else:
                pred_answer = float('inf')
        else:
            try:
                pred_answer = float(pred_answer)
            except ValueError as e:
                pred_answer = float('inf')
    return pred_answer

def LLMs_reasoning():
    apikey = args.apikey
    if args.correct == None:
        correct = 0
    else:
        correct = args.correct
    prompt_easy_cot, prompt_normal_cot, prompt_hard_cot = get_prompt()
    question, answer, ids = load_data(args)
    if os.path.exists("demos/{}_{}_test_set_diff.json".format(args.engine,args.dataset.lower())):
        with open("demos/{}_{}_test_set_diff.json".format(args.engine,args.dataset.lower()), "r") as f:
            zsc = json.load(f)
            zsc_syntax_complex = zsc["syntax_complex"]
            zsc_semantic_complex = zsc["semantic_complex"]
            zsc_difficulty = zsc["difficulty"]
            zsc_cot = zsc['cot']
    else:
        zsc = {
            "syntax_complex":[],
            "semantic_complex":[],
            "difficulty":[],
            "cot":[]
        }
        for idx, element in enumerate(question):
            print("{}/{}".format(idx + 1, len(question)))
            if args.dataset.lower() == 'aqua':
                inputs = prompt_easy_cot + '\n\nQ: ' + element.split('Answer Choices:')[0] + '\n' + 'Answer Choices:' + element.split('Answer Choices:')[1] + '\n' + 'A:'
            else:
                inputs = prompt_easy_cot + '\n\nQ: ' + element + '\n' + "A: "
            get_result = False
            while get_result == False:
                get_result, cot_answer, error_msg = basic_runner(args, inputs, 0, args.max_length_cot, apikey, n=1)
            zsc["cot"].append(cot_answer)
            zsc["syntax_complex"].append(len(cot_answer.split()))
            output_clean = text_cleanup(cot_answer, stopwords)
            output_clean = set(output_clean)
            input_clean = text_cleanup(element, stopwords)
            sum = 0
            for token in output_clean:
                if token not in input_clean:
                    sum += 1
            zsc["semantic_complex"].append(sum)
        zsc_difficulty = []
        for i in range(len(zsc["syntax_complex"])):
            zsc_difficulty.append(zsc["syntax_complex"][i] + zsc["semantic_complex"][i])
        zsc["difficulty"] = zsc_difficulty
        with open("demos/{}_{}_test_set_diff.json".format(args.engine, args.dataset.lower()), "w") as f:
            json.dump(zsc,f)
        zsc_syntax_complex = zsc['syntax_complex']
        zsc_semantic_complex = zsc["semantic_complex"]
        zsc_cot = zsc['cot']
    assert len(zsc_syntax_complex) == len(zsc_semantic_complex) == len(zsc_difficulty) == len(zsc_cot)

    if args.dataset in ["SVAMP","MultiArith","AddSub","SingleEq"]:
        dataset_ = "svamp"
    else:
        dataset_ = args.dataset.lower()

    if os.path.exists("demos/{}_{}_demo_set_diff_{}.json".format(args.engine,dataset_,args.sample_question_num)):
        with open("demos/{}_{}_demo_set_diff_{}.json".format(args.engine,dataset_,args.sample_question_num),'r') as f:
            zsc_corpus = json.load(f)
            zsc_corpus_syntax_complex = zsc_corpus["syntax_complex"]
            zsc_corpus_semantic_complex = zsc_corpus["semantic_complex"]
            zsc_corpus_difficulty = zsc_corpus["difficulty"]
            zsc_corpus_question = zsc_corpus["question"]
            zsc_corpus_cot = zsc_corpus["cot"]
    else:
        zsc_corpus = {
            "syntax_complex":[],
            "semantic_complex":[],
            "difficulty":[],
            "question":[],
            "cot":[]
        }
        unlabel_datas = load_unlabel_data(args)
        sample_question = random.sample(unlabel_datas, args.sample_question_num)
        zsc_corpus["question"] = sample_question
        for idx, element in enumerate(sample_question):
            print("{}/{}".format(idx + 1, len(sample_question)))
            if args.dataset.lower() == 'aqua':
                inputs = prompt_easy_cot + '\n\nQ: ' + element.split('Answer Choices:')[0] + '\n' + 'Answer Choices:' + element.split('Answer Choices:')[1] + '\n' + 'A:'
            else:
                inputs = prompt_easy_cot + '\n\nQ: ' + element + '\n' + "A: "
            get_result = False
            while get_result == False:
                get_result, cot_answer, error_msg = basic_runner(args, inputs, 0, args.max_length_cot, apikey, n=1)
            zsc_corpus['cot'].append(cot_answer)
            zsc_corpus["syntax_complex"].append(len(cot_answer.split()))
            output_clean = text_cleanup(cot_answer, stopwords)
            output_clean = set(output_clean)
            input_clean = text_cleanup(element, stopwords)
            sum = 0
            for token in output_clean:
                if token not in input_clean:
                    sum += 1
            zsc_corpus["semantic_complex"].append(sum)
        zsc_corpus_difficulty = []
        for i in range(len(zsc_corpus["syntax_complex"])):
            zsc_corpus_difficulty.append(zsc_corpus["syntax_complex"][i] + zsc_corpus["semantic_complex"][i])
        zsc_corpus["difficulty"] = zsc_corpus_difficulty
        zsc_corpus_syntax_complex = zsc_corpus['syntax_complex']
        zsc_corpus_semantic_complex = zsc_corpus['semantic_complex']
        zsc_corpus_question = zsc_corpus['question']
        zsc_corpus_cot = zsc_corpus['cot']
        with open("demos/{}_{}_demo_set_diff_{}.json".format(args.engine,dataset_,args.sample_question_num),'w') as f:
            json.dump(zsc_corpus, f)
    assert len(zsc_corpus_syntax_complex) == len(zsc_corpus_semantic_complex) == len(zsc_corpus_difficulty) == len(zsc_corpus_question) == len(zsc_corpus_cot)

    judge_all_temp = copy.deepcopy(zsc_corpus_difficulty)
    judge_all_temp.sort()
    v = np.std(zsc_corpus_difficulty) / np.mean(zsc_corpus_difficulty)
    v = v/2
    easy_threshold = judge_all_temp[int(len(judge_all_temp) * v)]
    normal_threshold = judge_all_temp[int(len(judge_all_temp) * (1-v))]
    corpus_diff = []
    for idx in range(len(zsc_corpus_difficulty)):
        if zsc_corpus_difficulty[idx] >= normal_threshold:
            corpus_diff.append('hard')
        elif zsc_corpus_difficulty[idx] >= easy_threshold:
            corpus_diff.append('normal')
        else:
            corpus_diff.append('easy')

    if os.path.exists("demos/{}_{}_fsc_{}.json".format(args.engine,dataset_,args.sample_question_num)):
        with open("demos/{}_{}_fsc_{}.json".format(args.engine,dataset_,args.sample_question_num), "r") as f:
            fsc = json.load(f)
            fsc_question = fsc["question"]
            fsc_cot = fsc["cot"]
    else:
        fsc = {
            "question":zsc_corpus_question,
            "cot":[]
        }
        for idx, element in enumerate(zsc_corpus_question):
            print("{}/{}".format(idx + 1, len(zsc_corpus_question)))
            if args.dataset.lower() == 'aqua':
                inputs_easy = prompt_easy_cot + '\n\nQ: ' + element.split('Answer Choices:')[0] + '\n' + 'Answer Choices:' + element.split('Answer Choices:')[1] + '\n' + 'A:'
                inputs_normal = prompt_normal_cot + '\n\nQ: ' + element.split('Answer Choices:')[0] + '\n' + 'Answer Choices:' + element.split('Answer Choices:')[1] + '\n' + 'A:'
                inputs_hard = prompt_hard_cot + '\n\nQ: ' + element.split('Answer Choices:')[0] + '\n' + 'Answer Choices:' + element.split('Answer Choices:')[1] + '\n' + 'A:'
            else:
                inputs_easy = prompt_easy_cot + '\n\nQ: ' + element + '\n' + "A: "
                inputs_normal = prompt_normal_cot + '\n\nQ: ' + element + '\n' + "A: "
                inputs_hard = prompt_hard_cot + '\n\nQ: ' + element + '\n' + "A: "
            if corpus_diff[idx] == 'hard':
                inputs = inputs_hard
            elif corpus_diff[idx] == 'normal':
                inputs = inputs_normal
            else:
                inputs = inputs_easy
            get_result, cot_answer, error_msg = basic_runner(args, inputs, 0, args.max_length_cot, apikey, n=1)

            if args.dataset.lower() == 'strategyqa':
                if "unknown" in cot_answer or "uncertain" in cot_answer:
                    fsc["cot"].append(None)
                    continue

            pred2 = None
            if 'answer is' in cot_answer:
                pred2 = cot_answer.split('answer is')[-1]
            elif 'answer choice is' in cot_answer:
                pred2 = cot_answer.split('answer choice is')[-1]
                cot_answer.replace('answer choice is', 'answer is')
            try:
                pred_answer = extract_answer(args, pred2)
            except:
                pred_answer = None
            if get_result and pred_answer:
                fsc["cot"].append(cot_answer)
            else:
                fsc["cot"].append(None)
        fsc_question = fsc["question"]
        fsc_cot = fsc["cot"]
        with open("demos/{}_{}_fsc_{}.json".format(args.engine,dataset_,args.sample_question_num), "w") as f:
            json.dump(fsc,f)
    assert len(fsc_question) == len(fsc_cot)

    #filter samples with fragile rationales
    for i, cot_answer in enumerate(fsc_cot):
        if cot_answer:
            cot_answer_list_split = cot_answer.replace(", ", "$$$$$").replace(". ", "$$$$$").replace("? ","$$$$$")\
                .replace("! ", "$$$$$").replace(",\n", "$$$$$").replace(".\n", "$$$$$").replace("?\n", "$$$$$")\
                .replace("!\n","$$$$$").split("$$$$$")
            if len(cot_answer_list_split) > 15:
                fsc_cot[i] = None

    question_pred = question[args.start:]
    for idx, element in enumerate(question_pred):
        idx = idx + args.start
        diff = zsc_difficulty[idx]
        if args.dataset.lower() == 'aqua':
            inputs_easy = prompt_easy_cot + '\n\nQ: ' + element.split('Answer Choices:')[0] + '\n' + 'Answer Choices:' + element.split('Answer Choices:')[1] + '\n' + 'A:'
            inputs_normal = prompt_normal_cot + '\n\nQ: ' + element.split('Answer Choices:')[0] + '\n' + 'Answer Choices:' + element.split('Answer Choices:')[1] + '\n' + 'A:'
            inputs_hard = prompt_hard_cot + '\n\nQ: ' + element.split('Answer Choices:')[0] + '\n' + 'Answer Choices:' + element.split('Answer Choices:')[1] + '\n' + 'A:'
        else:
            inputs_easy = prompt_easy_cot + '\n\nQ: ' + element + '\n' + "A: "
            inputs_normal = prompt_normal_cot + '\n\nQ: ' + element + '\n' + "A: "
            inputs_hard = prompt_hard_cot + '\n\nQ: ' + element + '\n' + "A: "
        if diff >= normal_threshold:
            diff_section = 'hard'
        elif diff >= easy_threshold:
            diff_section = 'normal'
        else:
            diff_section = 'easy'

        abs_difference = []
        for item in zsc_corpus_difficulty:
            abs_difference.append(abs(diff - item))
        index = nsmall1(args.dataset.lower(), args.sample_num, abs_difference, max(abs_difference) + 1, fsc_cot)
        inputs = ''
        for i in range(args.sample_num):
            inputs = inputs + "Q: " + fsc_question[index[i]] + '\n' + "A: " + fsc_cot[index[i]].replace('\n\n','\n') + '\n\n'
        if args.dataset.lower() == 'aqua':
            inputs = inputs + 'Q: ' + element.split('Answer Choices:')[0] + '\n' + 'Answer Choices:' + element.split('Answer Choices:')[1] + '\n' + 'A:'
        else:
            inputs = inputs + 'Q: ' + element + '\n' + "A: "

        pred_answer = None
        pred_for_save = None
        flag = 0
        repeat_times = 0
        while pred_answer == None:
            repeat_times += 1
            if repeat_times > 3:
                break
            get_result, pred, error_msg = basic_runner(args, inputs, flag, args.max_length_cot, args.apikey, n=1)
            pred_for_save = pred
            if not get_result:
                logger.warning(
                    f"not get predicted result (question id: {ids[idx]})."
                    f"ERROR Message: {error_msg if error_msg else None}"
                )
                #####
                if "Please reduce the length of the messages." in error_msg:
                    if zsc_difficulty[idx] >= normal_threshold:
                        inputs = inputs_hard
                    elif zsc_difficulty[idx] >= easy_threshold:
                        inputs = inputs_normal
                    else:
                        inputs = inputs_easy
                #####
                continue
            if 'answer is' in pred or 'answer choice is' in pred:
                pred2 = pred.split('answer is')[-1] if 'answer is' in pred else pred.split('answer choice is')[-1]
                try:
                    pred_answer = extract_answer(args, pred2)
                except:
                    pred_answer = None
                print("-")
                print(pred2)
            else:
                inputs2 = inputs + pred + ' ' + args.direct_answer_trigger_for_direct
                try:
                    get_result, pred3, error_msg = basic_runner(args, inputs2, 0, 32, args.apikey)
                    print('no answer:')
                    print(pred3)
                except Exception as e:
                    decode_error_data = {
                        'question': question[idx]
                    }
                    write_json(decode_error_data, Decoder_Error_File)
                    logger.warning(
                        f"an error raised when predicting (question id: {ids[idx]}). "
                        f"ERROR: {getattr(e.__class__, '__name__')}:{str(e)}"
                    )
                    continue
                if not get_result:
                    logger.warning(
                        f"not get predicted result (question id: {ids[idx]})."
                        f"ERROR Message: {error_msg if error_msg else None}"
                    )
                    if "Please reduce the length of the messages." in error_msg:
                        if zsc_difficulty[idx] >= normal_threshold:
                            inputs = inputs_hard
                        elif zsc_difficulty[idx] >= easy_threshold:
                            inputs = inputs_normal
                        else:
                            inputs = inputs_easy
                    continue
                try:
                    pred_answer = extract_answer(args, pred3)
                except:
                    pred_answer = None
            flag = 1

        ans = False
        if pred_answer is not None:
            if args.dataset.lower() in ["svamp", "gsm8k", "multiarith", "addsub", "singleeq"]:
                if abs(pred_answer - answer[idx]) <= 1e-3:
                    correct += 1
                    ans = True
                    json_data = {
                        "ID": ids[idx],
                        "question": question[idx],
                        "chain-of-thought": pred_for_save,
                        "pred": pred_answer,
                        "answer": answer[idx],
                        "difficulty": diff_section,
                        "ans": ans
                    }
                    write_json(json_data, Predict_File)
                else:
                    json_data = {
                        "ID": ids[idx],
                        "question": question[idx],
                        "chain-of-thought": pred_for_save,
                        "pred": pred_answer,
                        "answer": answer[idx],
                        "difficulty": diff_section,
                        "ans": ans
                    }
                    write_json(json_data, Predict_File)
            else:
                if isinstance(pred_answer, float) and isinstance(answer[idx], float):
                    precision = min(get_precision(pred_answer), get_precision(answer[idx]))
                    if round(pred_answer, precision) == round(answer[idx], precision):
                        correct += 1
                        ans = True
                        json_data = {
                            "ID": ids[idx],
                            "question": question[idx],
                            "chain-of-thought": pred_for_save,
                            "pred": pred_answer,
                            "answer": answer[idx],
                            "difficulty": diff_section,
                            "ans": ans
                        }
                        write_json(json_data, Predict_File)
                    else:
                        ans = False
                        json_data = {
                            "ID": ids[idx],
                            "question": question[idx],
                            "chain-of-thought": pred_for_save,
                            "pred": pred_answer,
                            "answer": answer[idx],
                            "difficulty": diff_section,
                            "ans": ans
                        }
                        write_json(json_data, Predict_File)
                else:
                    if pred_answer == answer[idx]:
                        correct += 1
                        ans = True
                        json_data = {
                            "ID": ids[idx],
                            "question": question[idx],
                            "chain-of-thought": pred_for_save,
                            "pred": pred_answer,
                            "answer": answer[idx],
                            "difficulty":diff_section,
                            "ans": ans
                        }
                        write_json(json_data, Predict_File)
                    else:
                        json_data = {
                            "ID": ids[idx],
                            "question": question[idx],
                            "chain-of-thought": pred_for_save,
                            "pred": pred_answer,
                            "answer": answer[idx],
                            "difficulty": diff_section,
                            "ans": ans
                        }
                        write_json(json_data, Predict_File)
        else:
            json_data = {
                "ID": ids[idx],
                "question": question[idx],
                "chain-of-thought": pred_for_save,
                "pred": pred_answer,
                "answer": answer[idx],
                "difficulty": diff_section,
                "ans": ans
            }
            write_json(json_data, Predict_File)
        json_data = {"correct": correct, 'tested': idx + 1, "Acc": correct / (idx + 1)}
        write_json(json_data, Predict_File)
        if idx >= args.end - 1:
            break
    return correct

def print_exp(args, return_flag=0):
    info = ''
    for k, v in vars(args).items():
        info += '{}:{}\n'.format(k, v)
    print('---------------experiment args---------------')
    print(info)
    print('---------------------------------------------')
    if return_flag == 0:
        return
    elif return_flag == 1:
        return info
    else:
        pass

if __name__ == '__main__':
    sample_num_dcit = {"AQuA": 4, "gsm8k": 8, "SVAMP": 8, "AddSub":8, "MultiArith":8, "SingleEq":8, "last_letters":4, "coin_flip":8, "CommonsenseQA":7, "StrategyQA":6}
    args.sample_num = sample_num_dcit[args.dataset]
    print_exp(args)
    if args.correct == None and args.tested == None:
        write_json(vars(args),Predict_File)
    alls = LLMs_reasoning()
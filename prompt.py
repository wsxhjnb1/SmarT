import json

from config import args

Demo_Folder = 'demos/'
def get_prompt():
    if args.task == 'math':
        if args.dataset.lower() == 'aqua':
            demo_easy_file = Demo_Folder + 'aqua_easy.txt'
            demo_normal_file = Demo_Folder + 'aqua_normal.txt'
            demo_hard_file = Demo_Folder + 'aqua_hard.txt'
        else:
            demo_easy_file = Demo_Folder + 'gsm8k_easy.txt'
            demo_normal_file = Demo_Folder + 'gsm8k_normal.txt'
            demo_hard_file = Demo_Folder + 'gsm8k_hard.txt'
    elif args.task == 'commonsense':
        if args.dataset.lower() == 'commonsenseqa':
            demo_easy_file = Demo_Folder + 'commonsenseqa_easy.txt'
            demo_normal_file = Demo_Folder + 'commonsenseqa_normal.txt'
            demo_hard_file = Demo_Folder + 'commonsenseqa_hard.txt'
        else:
            demo_easy_file = Demo_Folder + 'strategyqa_easy.txt'
            demo_normal_file = Demo_Folder + 'strategyqa_normal.txt'
            demo_hard_file = Demo_Folder + 'strategyqa_hard.txt'
    elif args.task == 'symbolic':
        if args.dataset.lower() == 'last_letters':
            demo_easy_file = Demo_Folder + 'last_letters_easy.txt'
            demo_normal_file = Demo_Folder + 'last_letters_normal.txt'
            demo_hard_file = Demo_Folder + 'last_letters_hard.txt'
        else:
            demo_easy_file = Demo_Folder + 'coin_flip_easy.txt'
            demo_normal_file = Demo_Folder + 'coin_flip_normal.txt'
            demo_hard_file = Demo_Folder + 'coin_flip_hard.txt'

    with open(demo_easy_file, "r", encoding='utf-8') as f:
        prompt_easy_cot = f.read().strip()
    with open(demo_normal_file, "r", encoding='utf-8') as f:
        prompt_normal_cot = f.read().strip()
    with open(demo_hard_file, "r", encoding='utf-8') as f:
        prompt_hard_cot = f.read().strip()
    return prompt_easy_cot, prompt_normal_cot, prompt_hard_cot
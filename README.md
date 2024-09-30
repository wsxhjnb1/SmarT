## Adaption-of-Thought: Learning Question Difficulty Improves Large Language Models for Reasoning
Code and data of our paper "Adaption-of-Thought: Learning Question Difficulty Improves Large Language Models for Reasoning" (EMNLP 2024)

## 1. Content
* dataset folder: Containing the source data for inference and the code for building the unlabeled question corpus.
* demos folder: Containing all demonstrations used in our paper.
* config.py: Setting the parameter for our method.
* gpt_runner.py: Program used to call the gpt family models.
* llama_runner.py: Program used to call the llama family models.
* main_gpt.py: Adopting the gpt family model for reasoning.
* main_llama.py: Adopting the llama family model for reasoning.
* prompt.py: Reading the demonstrations in the demos folder.
* requirements.txt: The environment required to run this repository.
* utils.py: Programs used to read and write data.

## 2. Environment reuqirement
```
pip install -r requirements.txt
```

## 3. Quickly start on GPT
```
# Please set apikey at config.py before reasoning

python main_gpt.py --task math --dataset AQuA
python main_gpt.py --task math --dataset gsm8k
python main_gpt.py --task math --dataset SVAMP
python main_gpt.py --task math --dataset AddSub
python main_gpt.py --task math --dataset MultiArith
python main_gpt.py --task math --dataset SingleEq

python main_gpt.py --task symbolic --dataset last_letters
python main_gpt.py --task symbolic --dataset coin_flip

python main_gpt.py --task commonsense --dataset CommonsenseQA
python main_gpt.py --task commonsense --dataset StrategyQA
```

## 4. Quickly start on llama2
```
# Please set llama path at main_llama.py before reasoning

# set engine as llama2_7b_chat
python main_llama.py --task math --dataset AQuA --engine 'llama2_7b_chat'

# set engine as llama2_13b_chat
python main_llama.py --task math --dataset AQuA --engine 'llama2_13b_chat'

# set engine as llama2_70b_chat
python main_llama.py --task math --dataset AQuA --engine 'llama2_70b_chat'
```

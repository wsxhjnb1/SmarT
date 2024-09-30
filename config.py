import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apikey", type=str, default=""
    )
    parser.add_argument(
        "--mode", type=str, default='complex-cot'
    )
    parser.add_argument(
        "--sample_num", type=int, default=4
    )
    parser.add_argument(
        "--sample_question_num", type=int, default=50
    )
    parser.add_argument(
        "--task", type=str, default='math',choices=['math','commonsense','symbolic']
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=4097,
        help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32,
        help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=3.0, help=""
    )
    parser.add_argument(
        "--temperature", type=float, default=1, help=""
    )
    parser.add_argument(
        '--dataset', default='AQuA',
        help="dataset",
        choices=["SVAMP", "gsm8k", "AQuA", "MultiArith", "AddSub", "SingleEq", "CommonsenseQA", "coin_flip",
                 "last_letters", "FinQA", "TATQA", "ConvFinQA", "StrategyQA"]
    )
    parser.add_argument(
        "--engine", default='gpt-3.5-turbo-0613',
        choices=['gpt-3.5-turbo-0613','gpt-3.5-turbo','llama2_7b_chat','llama2_13_chat','llama2_70b_chat']
    )
    parser.add_argument(
        '--correct', default=0, type=int
    )
    parser.add_argument(
        "--test_start", default='0', help='string, number'
    )
    parser.add_argument(
        "--test_end", default='full', help='string, number'
    )
    parser.add_argument(
        "--start", default='0', type=int, help='string, number'
    )
    parser.add_argument(
        "--end", default='9999999', type=int, help='string, number'
    )
    parser.add_argument(
        '--answer_extracting_prompt', default='Therefore, the answer is', type=str
    )
    parser.add_argument(
        "--thread", default=10, type=int,
    )
    parsed_args = parser.parse_args()
    parsed_args.direct_answer_trigger_for_zeroshot = "Let's think step by step."
    parsed_args.direct_answer_trigger_for_direct = "Therefore, the answer is"
    return parsed_args


args = parse_arguments()

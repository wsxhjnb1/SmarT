from vllm import SamplingParams
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def llama_runner_chat(llm, args, inputs_all, temperature, max_length_cot, n=1, stop="\n\nQ:"):
    temperature=temperature

    # 定义采样参数，temperature 控制生成文本的多样性，top_p 控制核心采样的概率
    sampling_params = SamplingParams(temperature=temperature, top_p=1, max_tokens=max_length_cot, n=n,stop=stop)

    prompts_re_generate=inputs_all
    preds = []
    # 只能批次化生成后再解析
    outputs_re_generate = llm.generate(prompts_re_generate, sampling_params)
    if n == 1:
        for test_idx, output in enumerate(outputs_re_generate):
            prompt = output.prompt  # 获取原始的输入提示
            pred = output.outputs[0].text # 从输出对象中获取生成的文本
            preds.append(pred)
    else:
        for test_idx, output in enumerate(outputs_re_generate):
            pred_ = []
            for idx_n in range(n):
                pred = output.outputs[idx_n].text # 从输出对象中获取生成的文本
                pred_.append(pred)
            preds.append(pred_)
    return preds
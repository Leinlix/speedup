# Use a pipeline as a high-level helper
import pytest
from modelscope import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from speedup.compiler.compiler import compile_model
from speedup.compiler.compile_config import CompileConfig


model_name ="Qwen/Qwen3-0.6B"
prompt = "Tell me something about large language model"
messages = [
    {
        "role": "system",
        'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
    },
    {
        'role': 'user',
        'content': prompt
    }
]
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)

def test_modelscope():
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto')

    model_input = tokenizer([text], return_tensors='pt').to(model.device)

    generated_ids = model.generate(**model_input, max_new_tokens=32768)

    output_ids = generated_ids[0][len(model_input.input_ids[0]):].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print('thinking content:', thinking_content)
    print('content:', content)


def test_vllm():
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=32768)
    llm = LLM(model=model_name)
    outputs = llm.generate([text], sampling_params)

    for outputs in outputs:
        generated_text = outputs.outputs[0].text
        print(generated_text)

def test_speedup():
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=32768)
    llm = LLM(model=model_name)
    # compile model
    model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
    ori_model = model_runner.model
    cp = CompileConfig(verbose = True)
    compiled_model = compile_model(ori_model, cp)
    model_runner.model = compiled_model
    outputs = llm.generate([text], sampling_params)

    for outputs in outputs:
        generated_text = outputs.outputs[0].text
        print(generated_text)

test_speedup()
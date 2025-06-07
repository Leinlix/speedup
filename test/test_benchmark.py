# Use a pipeline as a high-level helper
import pytest
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name ="Qwen/Qwen3-0.6b"
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

def test_modelscope():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto')

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
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

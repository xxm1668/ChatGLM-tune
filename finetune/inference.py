# !/usr/bin/env python3
"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

inference 训练好的模型。

Author: pankeyu
Date: 2023/03/17
"""

import torch
from peft import get_peft_model, LoraConfig, TaskType

from transformers import AutoTokenizer
from modeling_chatglm import ChatGLMForConditionalGeneration

torch.set_default_tensor_type(torch.cuda.HalfTensor)
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/znn/chatglm-model/chatglm-6b_model", trust_remote_code=True)
model = ChatGLMForConditionalGeneration.from_pretrained("/root/autodl-tmp/znn/chatglm-model/chatglm-6b_model",
                                                        trust_remote_code=True, device_map='auto')

peft_path = "checkpoints/model_best/chatglm-lora.pt"
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

max_seq_len = 400
model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(peft_path), strict=False)


def inference(
        instuction: str,
        sentence: str
):
    """
    模型 inference 函数。

    Args:
        instuction (str): _description_
        sentence (str): _description_

    Returns:
        _type_: _description_
    """
    with torch.no_grad():
        input_text = f"Instruction: {instuction}\n"
        if sentence:
            input_text += f"Input: {sentence}\n"
        input_text += f"Answer: "
        batch = tokenizer(input_text, return_tensors="pt")
        out = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=torch.ones_like(batch["input_ids"]).bool(),
            max_length=max_seq_len,
            temperature=0
        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.split('Answer: ')[-1]
        return answer


if __name__ == '__main__':
    from rich import print

    samples = [
        {
            'instruction': "你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。",
            "input": "帮我提取出下面句子中所有的SPO，并输出为json，不要做多余的回复：\n\n5  年，徐舜寿考入清华大学机械工程系航空工程组\nAnswer:",
        },
        {
            'instruction': "你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。",
            "input": "抽取下列句子中包含的所有关系，必须用json形式返回结果，除此之外不要说任何其他话。\n\n《江湖情长》是落木习习在17K连载的武侠小说，故事讲述男主角夏云洛为报父母之仇而勇闯江湖，其间遇到各种各样的人，与各路美女之间的爱恨纠葛\nAnswer:",
        }
    ]

    for sample in samples:
        print(sample['input'].split('\n\n')[1])
        res = inference(
            sample['instruction'],
            sample['input']
        )
        print(res)

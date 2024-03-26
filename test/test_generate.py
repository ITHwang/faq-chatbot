import os

import torch

from model import FAQModel
from transformers import AutoTokenizer

def prompt(question):
    return f"질문: {question}\n답변: "

model_name = "beomi/llama-2-ko-7b"
max_length = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stream = True

model = FAQModel(model_name, device)

question = "나 도둑맞았어 이제 어떻게 해야 해? 번호 매겨서 간단 명료하게 적어줘."
prompt_text = prompt(question)
answer = model.answer(prompt_text, max_length, stream=stream)

# print(answer)
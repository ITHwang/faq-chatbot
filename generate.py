import os

import torch

from model import QuestionAnsweringModel

model_name = "beomi/KoAlpaca-KoRWKV-1.5B"
max_length = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = QuestionAnsweringModel(model_name)

question = "나 도둑맞았어 이제 어떻게 해야 해?"

output_text = model.answer(question, max_length=max_length, device=device)

print(output_text)

    




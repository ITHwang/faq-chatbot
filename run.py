import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
import json

import chromadb
import torch

from model import FAQEmbedding, FAQModel
from utils.function import ask_continue
from utils.dto import QA
from utils.example_selector import ExampleSelector
from utils.memory import Memory
from utils.prompt import PromptTemplate

BASE_DIR = Path(__file__).parent
db_path = str(BASE_DIR / "data" / "output")
json_path = str(BASE_DIR / "data" / "output" / "faq.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "beomi/llama-2-ko-7b"
max_gen_len = 550
stream = True
temperature: float = (0.5,)
top_p: float = (0.9,)
echo: bool = (False,)

topk = 3
n_history = 3
memory_size = 5
cannot_answer_comment = (
    "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다."
)
thres = 0.165

print("loading models...", flush=True)
embedding_model = FAQEmbedding()
llm = FAQModel(model_name, device)
print("loaded", flush=True)
example_selector = ExampleSelector.from_examples(json_path, db_path, embedding_model)
memory = Memory(memory_size)

prompt_template = PromptTemplate(example_selector, memory, cannot_answer_comment)

greeting = """
안녕하세요! 네이버 스마트스토어 고객센터 챗봇입니다.
빠르고 정확한 상담을 위해 이름과 주민등록번호 앞자리를 입력해주세요.
"""
print(greeting)

client_name = input("이름: ")
client_id = input("주민등록번호 앞자리(ex. 980428): ")

help = """
고객 확인 완료되었습니다. 무엇을 도와드릴까요?
"""
print(help)

while True:
    query = input("질문: ").strip()
    prompt_text = prompt_template.prompt(query, topk=topk, n_history=n_history, thres=thres)

    if prompt_text == cannot_answer_comment:
        print(cannot_answer_comment)
        continue

    print(prompt_text)

    answer = llm.answer(prompt_text, max_gen_len, stream=stream)
    memory.push(QA(query, answer.strip()))

    print(memory)

    if not ask_continue():
        break

ending = """
네이버 스마트스토어 고객센터 챗봇을 이용해주셔서 감사합니다.
"""
print(ending)

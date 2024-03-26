import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
import json

import chromadb

from model import FAQEmbedding

BASE_DIR = Path(__file__).parent.parent
db_path = str(BASE_DIR / "data" / "output")
json_path = str(BASE_DIR / "data" / "output/faq.json")
topk = 100

with open(json_path, "r") as file:
    dic = json.load(file)
faqs = [(int(e["id"]), e["question"], e["answer"]) for e in dic]
faqs.sort(key=lambda x: x[0])

client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name="faq", embedding_function=FAQEmbedding())

while True:
    question = input("질문을 입력해주세요: ")

    results = collection.query(
        # Note "query: ": a prefix rule of E5 embedding model.
        query_texts=[f"query: {question}"],
        n_results=topk,
    )
    ids, dists = results["ids"][0], results["distances"][0]

    with open("a.txt", "w") as file:
        for id_, dist in zip(ids, dists):
            file.write(f"Distance: {dist}\n")
            file.write(f"Question: {faqs[int(id_)][1]}\n")
            file.write(f"Answer: {faqs[int(id_)][2]}\n\n\n")

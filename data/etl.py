import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
import pickle
import tqdm

from yattag import Doc
import chromadb
from model import FAQEmbedding

import utils.function as function

BASE_DIR = Path(__file__).parent.parent
file_path = str(BASE_DIR / "data" / "final_result.pkl")
db_path = str(BASE_DIR / "data" / "output")
json_path = str(BASE_DIR / "data" / "output/faq.json")

print("loading faq data...", flush=True)

with open(file_path, "rb") as file:
    dic = pickle.load(file)


faqs = []
for i, (question, answer) in enumerate(tqdm.tqdm(dic.items())):
    faq = {}

    cut_position = answer.find("위 도움말이 도움이 되었나요?")
    if cut_position != -1:
        faq["answer"] = answer[:cut_position].strip()

    parts = question.strip().split("]")
    faq["categories"] = [p.replace("[", "").strip() for p in parts[:-1]]
    faq["question"] = parts[-1].strip()

    doc, tag, text = Doc().tagtext()
    with tag("div", klass="question-and-answer"):
        with tag(
            "div",
            klass="question",
            **{f"category{i+1}": cat for i, cat in enumerate(faq["categories"])},
        ):
            text(question)
        with tag(
            "div",
            klass="answer",
            **{f"category{i+1}": cat for i, cat in enumerate(faq["categories"])},
        ):
            text(answer)
    faq["html"] = doc.getvalue()

    doc, tag, text = Doc().tagtext()
    with tag(
        "div",
        klass="question",
        **{f"category{i+1}": cat for i, cat in enumerate(faq["categories"])},
    ):
        text(question)
    faq["question_html"] = doc.getvalue()

    faqs.append(faq)

print("loaded faq data", flush=True)

print(f"Total doc: {len(faqs)}")

lower_bound, upper_bound = function.get_outlier_bound([len(faq["html"]) for faq in faqs])
lower_outliers = [faq for faq in faqs if len(faq["html"]) < lower_bound]
uppper_outliers = [faq for faq in faqs if len(faq["html"]) > upper_bound]
print(f"The number of too short doc: {len(lower_outliers)}")
print(f"The number of too long doc: {len(uppper_outliers)}")

faqs = [faq for faq in faqs if len(faq["html"]) >= lower_bound and len(faq["html"]) <= upper_bound]
print(f"Total doc after removing outliers: {len(faqs)}")
print(f"min length: {min(len(faq['html']) for faq in faqs)}")
print(f"max length: {max(len(faq['html']) for faq in faqs)}")
print(f"mean length: {int(sum(len(faq['html']) for faq in faqs) / len(faqs))}")
print(f"mean length of answer: {int(sum(len(faq['answer']) for faq in faqs) / len(faqs))}")

print("saving faq data...", flush=True)

client = chromadb.PersistentClient(path=db_path)
# client.delete_collection(name="faq")
collection = client.create_collection(
    name="faq",
    metadata={"hnsw:space": "cosine"},
    embedding_function=FAQEmbedding(),
)
# collection = client.get_collection(name="faq", embedding_function=FAQEmbedding())

for i, faq in enumerate(tqdm.tqdm(faqs)):
    # faq["id"] = str(i)py

    # Note "passage: ": a prefix rule of E5 embedding model.
    collection.add(documents=[f"passage: {faq['html']}"], ids=[str(i)])

# with open(json_path, 'w', encoding='utf-8') as json_file:
#     json.dump(faqs, json_file, indent=4, ensure_ascii=False)

print(f"saved {collection.count()} documents", flush=True)

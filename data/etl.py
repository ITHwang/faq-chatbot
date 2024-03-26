import argparse
import os
import sys
from pathlib import Path
import pickle
import tqdm
import json

from yattag import Doc
import chromadb

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model import FAQEmbedding
import utils.function as function

BASE_DIR = Path(__file__).parent
db_path = BASE_DIR / "db"
if not db_path.exists():
    BASE_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(description="Process some file paths.")

parser.add_argument(
    "--pkl_path",
    type=str,
    help="The path to the pkl file",
)

parser.add_argument("--db_path", type=str, default=str(db_path), help="The path to the DB file")

parser.add_argument(
    "--json_path",
    type=str,
    default=str(BASE_DIR / "faq.json"),
    help="The path to the JSON file",
)

args = parser.parse_args()

pkl_path = args.pkl_path
db_path = args.db_path
json_path = args.json_path

# load raw data
with open(pkl_path, "rb") as file:
    dic = pickle.load(file)

# convert raw data to faq data
faqs = []
for i, (question, answer) in enumerate(tqdm.tqdm(dic.items())):
    faq = {}

    # remove unnecessary parts
    cut_position = answer.find("위 도움말이 도움이 되었나요?")  # ending message
    if cut_position != -1:
        faq["answer"] = answer[:cut_position].strip()

    # parse categories, question
    parts = question.strip().split("]")
    faq["categories"] = [p.replace("[", "").strip() for p in parts[:-1]]
    faq["question"] = parts[-1].strip()

    # build html for each faq using yattag
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

    faqs.append(faq)

print("loaded faq data", flush=True)

print(f"Total doc: {len(faqs)}")

# remove outliers
lower_bound, upper_bound = function.get_outlier_bound([len(faq["html"]) for faq in faqs])
lower_outliers = [faq for faq in faqs if len(faq["html"]) < lower_bound]
uppper_outliers = [faq for faq in faqs if len(faq["html"]) > upper_bound]
print(f"The number of too short doc: {len(lower_outliers)}")  # 0
print(f"The number of too long doc: {len(uppper_outliers)}")  # about 200
faqs = [faq for faq in faqs if len(faq["html"]) >= lower_bound and len(faq["html"]) <= upper_bound]

# stat
print(f"Total doc after removing outliers: {len(faqs)}")
print(f"min length: {min(len(faq['html']) for faq in faqs)}")
print(f"max length: {max(len(faq['html']) for faq in faqs)}")
print(f"mean length: {int(sum(len(faq['html']) for faq in faqs) / len(faqs))}")  # about 550
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
    faq["id"] = str(i)

    # save faq vector data
    # Note "passage: ": a prefix rule of E5 embedding model.
    collection.add(documents=[f"passage: {faq['html']}"], ids=[str(i)])

# save faq meta data
with open(json_path, "w", encoding="utf-8") as json_file:
    json.dump(faqs, json_file, indent=4, ensure_ascii=False)

print(f"saved {collection.count()} documents", flush=True)

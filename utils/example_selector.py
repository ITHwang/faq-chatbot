import os
import sys
from pathlib import Path
import json

import chromadb

import model
from utils.dto import QA


class ExampleSelector:
    """
    like langchain.prompts.example_selector
    """

    def __init__(self, examples: str, db_client: str, embedding_model):
        self.examples = examples
        self.db_client = db_client
        self.embedding_model = embedding_model

    @classmethod
    def from_examples(cls, example_path: str, db_path: str, embedding_model):
        examples = cls.load_example(example_path)
        db_client = chromadb.PersistentClient(path=db_path)
        embedding_model = embedding_model

        return cls(examples, db_client, embedding_model)

    @staticmethod
    def load_example(example_path: str):
        """
        load qa meta data
        """
        with open(example_path, "r") as file:
            dic = json.load(file)
        qas = [(int(e["id"]), QA(e["question"], e["answer"])) for e in dic]
        qas.sort(key=lambda x: x[0])
        return qas

    def select(self, query, collection_name="faq", topk=10, thres=0.165):
        collection = self.db_client.get_collection(
            name=collection_name, embedding_function=self.embedding_model
        )

        results = collection.query(
            # Note "query: ": a prefix rule of E5 embedding model.
            query_texts=[f"query: {query}"],
            n_results=topk,
        )
        ids, dists = results["ids"][0], results["distances"][0]
        ids = [int(id_) for id_, dist in zip(ids, dists) if dist < thres]

        # get examples: if ids is empty, the query is not related to the domain
        return [self.examples[id_][1] for id_ in ids] if ids else []

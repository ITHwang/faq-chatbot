from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from chromadb import Documents, EmbeddingFunction, Embeddings


class FAQEmbedding(EmbeddingFunction):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
        self.model = AutoModel.from_pretrained("intfloat/multilingual-e5-large").to("cuda")

    def __call__(self, input: Documents) -> Embeddings:
        return [self.embed(doc) for doc in input]

    @torch.inference_mode()
    def embed(self, text: List[str]) -> Sequence[float]:
        batch_dict = self.tokenizer(
            text,
            max_length=512,
            stride=128,
            padding=True,
            truncation=True,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = batch_dict["input_ids"].to("cuda"), batch_dict[
            "attention_mask"
        ].to("cuda")

        outputs = self.model(input_ids, attention_mask=attention_mask)

        embedding = average_pool(outputs.last_hidden_state, attention_mask)
        if embedding.size()[0] > 1:
            embedding = embedding.sum(dim=0, keepdim=True) / embedding.size()[0]

        embedding = F.normalize(embedding, p=2, dim=1).to("cpu")

        return embedding.squeeze_().tolist()


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

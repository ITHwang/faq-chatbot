from typing import List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch import nn
import torch.functional as F
from transformers import LlamaForCausalLM, AutoTokenizer


class FAQModel(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.model = LlamaForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.5,
        top_p: float = 0.9,
        echo: bool = False,
        stream: bool = False,
    ) -> Tuple[List[List[int]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.
            stream (bool, optional): Flag indicating whether to generate text in a streaming manner. Defaults to False.

        Returns:
            Tuple[List[List[int]]]: A tuple containing generated token sequences

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
        """
        bsz = len(prompt_tokens)
        max_input_len = self.model.config.max_position_embeddings  # 2048

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)

        assert max_prompt_len <= max_input_len
        total_len = min(max_input_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_token_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.device)
        # input prompt tokens
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = (
                torch.tensor(t, dtype=torch.long, device=self.device).clone().detach()
            )

        input_text_mask = tokens != pad_id

        # for early stopping when generating zero white space token(\u200b) continuously
        n_zero_white_space = 0
        for cur_pos in range(min_prompt_len, total_len):
            output = self.model.forward(tokens[:, :cur_pos], input_text_mask)

            if temperature > 0:
                probs = torch.softmax(output.logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(output.logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            if next_token.item() == 30166:  # \u200b
                n_zero_white_space += 1
                if n_zero_white_space > 10:  # Note 10: hard coding
                    break
            else:
                n_zero_white_space = 0

            # whether to print generated text in streaming manner
            if stream:
                out_tok = self.tokenizer.decode(next_token)
                print(out_tok, end="", flush=True)

            tokens[:, cur_pos] = next_token
            input_text_mask[:, cur_pos] = torch.tensor(True, dtype=torch.bool)

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start:]
            out_tokens.append(toks)

        if stream:
            print()

        return out_tokens

    def answer(self, prompt_text: str, max_length: int, stream: bool = False):
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        outputs = self.generate(inputs.input_ids, max_gen_len=max_length, stream=stream)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

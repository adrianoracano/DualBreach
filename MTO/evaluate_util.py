"""Response + judge helpers used by MTO (`attack_suffix.py`, `decode`).

The upstream release does not ship this module; this is a reconstruction from how it is called:
  get_model_inference_pipeline(model_id) -> chat text-generation pipeline
  evaluate_llm_response(response_pipe, judge_pipe, judge_tokenizer, prompt) -> (jailbreak_score, keyword_score, response)
The jailbreak score is the one of eval/response_eval/utils.py (keyword refusal check + LLM judge, 0.5 / 0.5).
"""
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval", "response_eval"))

MAX_NEW_TOKENS = 512


class _LazyPipeline:
    """Chat text-generation pipeline that is loaded on first use: most runs never get past the guard, so the response
    and judge models would otherwise sit idle on the GPU. Attribute access and calls are forwarded to the pipeline."""

    def __init__(self, model_id, device):
        self.model_id, self.device, self._pipe = model_id, device, None

    def _load(self):
        if self._pipe is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16, device_map={"": self.device})
            self._pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=self.device)
        return self._pipe

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._load(), name)


_PIPELINES = {}


def get_model_inference_pipeline(model_id, device="cuda:2"):
    """One lazily-loaded pipeline per (model, device), shared across samples."""
    if (model_id, device) not in _PIPELINES:
        _PIPELINES[(model_id, device)] = _LazyPipeline(model_id, device)
    return _PIPELINES[(model_id, device)]


def _respond(response_pipe, prompt):
    if isinstance(response_pipe, str):  # name of an API model
        from utils import LLM
        return LLM(prompt, response_pipe) or ""
    text = response_pipe.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        enable_thinking=False)  # ignored by templates without a thinking mode
    out = response_pipe(text, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, return_full_text=False)
    return out[0]["generated_text"]


def evaluate_llm_response(response_pipe, judge_pipe, judge_tokenizer, prompt):
    from utils import get_jailbreak_score
    response = _respond(response_pipe, prompt)
    score, keyword_score, _ = get_jailbreak_score(judge_pipe, prompt, response, judge_tokenizer)
    return score, keyword_score, response


def harmful_paraphrase(original_prompt):
    """Scenario-style rewrite of the query by the local GGUF paraphraser (CPU, ~1-2 min); the original prompt is kept
    if no valid rewrite comes back so that a run is never aborted by it."""
    import local_paraphraser
    try:
        return local_paraphraser.harmful_paraphrase(original_prompt)
    except RuntimeError as e:
        print(f"harmful_paraphrase failed ({e}); keeping the original prompt")
        return original_prompt

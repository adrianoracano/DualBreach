#!/usr/bin/env python
# coding: utf-8

import os
import nltk_compat  # noqa: F401  (offline fallbacks for NLTK data; must precede other nltk imports)
import numpy as np
import time
import wandb
import argparse
import random
import sys
sys.path.insert(0, './GPT2ForwardBackward')

# from nltk import data

# data.path.append("/data/home/Xinzhe/COLD-Attack")

from nltk.corpus import stopwords
from opt_util import *
from util import *
from bleuloss import batch_log_bleulosscnn_ae
import torch.nn as nn


stop_words = set(stopwords.words('english'))


# class DNNClassifier(nn.Module):
#     def __init__(self):
#         super(DNNClassifier, self).__init__()
#         self.fc1 = nn.Linear(4096, 1024)
#         self.fc2 = nn.Linear(1024, 256)
#         self.fc3 = nn.Linear(256, 64)
#         # self.fc4 = nn.Linear(64, 16)
#         # self.fc5 = nn.Linear(16, 11)  # 11 categories corresponding to OpenAI Moderation API
#         self.fc45 = nn.Linear(64, 11)  # 11 categories corresponding to OpenAI Moderation API
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.5)
        
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc3(x))
#         # x = self.relu(self.fc4(x))
#         # x = self.dropout(x)
#         # x = self.fc5(x)
        
#         x = self.dropout(x)
#         x = self.fc45(x)
#         return x


class DNNClassifier(nn.Module):
    def __init__(self):
        super(DNNClassifier, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))         
        x = self.relu(self.fc2(x))  
        x = self.dropout(x)
        x = self.relu(self.fc3(x))  
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        
        return x


def options():
    parser = argparse.ArgumentParser()
    ## setting
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--no-cuda", action="store_true", help="no cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--run_tag", type=str, default="", help="suffix for the results jsonl name, to keep runs apart")
    parser.add_argument("--guard_loss", type=str, default="ce_prob", choices=["ce_prob", "margin"],
                        help="d3 guard loss: cross-entropy on [P(safe), P(unsafe)] (paper) or the logit margin unsafe - safe")
    parser.add_argument("--guard_align", type=str, default="retok", choices=["retok", "ids"],
                        help="guard input: the decoded suffix re-tokenised with the prompt (paper; the gradient rows are then "
                             "taken from the last len(suffix) tokens) or the exact suffix ids at known positions "
                             "(needs a target that shares d3's vocabulary, i.e. Qwen3.5)")
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--pretrained_model", type=str, default="llama2")
    parser.add_argument("--pretrained_judge_model", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--response_device", type=str, default="cuda:2", help="device of the victim/response model (loaded lazily, on first guard pass)")
    parser.add_argument("--judge_device", type=str, default="cuda:3", help="device of the LLM judge (loaded lazily, on first guard pass)")
    parser.add_argument("--pretrained_guard_model", type = str, default = "None")
    parser.add_argument("--dataset", type=str, default="advbench_original",
                        help="dataset_flag of attack_suffix.py (advbench_original, DNA_original, harmBench_original)")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--straight-through", action="store_true")
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--rl-topk", type=int, default=0)
    parser.add_argument("--lexical", type=str, default='max', choices=['max', 'ppl_max', 'all', 'bleu'])
    parser.add_argument("--lexical-variants", action="store_true", help="")
    parser.add_argument("--if-zx", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--guard", action="store_true")
    parser.add_argument("--device-id", type = int, default = 0, help = "device id for generating adversarial examples")
    parser.add_argument('--guard-device-id', type=int, default=1, help='device id for guarding model')
    ## experiment
    parser.add_argument("--input-file", type=str,
                        default="./data/lexical/commongen_data/test.multi.constraint.json")
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--start", type=int, default=1, help="loading data from ith examples.")
    parser.add_argument("--end", type=int, default=10, help="loading data util ith examples.")
    parser.add_argument("--repeat-batch", type=int, default=1, help="loading data util ith examples.")
    parser.add_argument("--mode", type=str, default='constrained_langevin',
                        choices=['suffix', 'control', 'paraphrase'])
    parser.add_argument("--control-type", type=str, default='sentiment', choices=['sentiment', 'lexical', 'style', 'format'])
    ## model
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, default=15, help="maximum length of optimized logits.")
    parser.add_argument("--max-length", type=int, default=50, help="maximum length of complete sentence.")
    parser.add_argument("--frozen-length", type=int, default=0, help="length of optimization window in sequence.")
    parser.add_argument("--goal-weight", type=float, default=0.1)
    parser.add_argument("--rej-weight", type=float, default=0.05)
    parser.add_argument("--abductive-filterx", action="store_true", help="filter out keywords included in x")
    parser.add_argument("--lr-nll-portion", type=float, default=1)
    parser.add_argument("--prefix-length", type=int, default=0, help="length of prefix.")
    parser.add_argument("--counterfactual-max-ngram", type=int, default=3)
    parser.add_argument("--no-loss-rerank", action="store_true", help="")
    parser.add_argument("--use-sysprompt", action="store_true", help="")
    # temperature
    parser.add_argument("--input-lgt-temp", type=float, default=1,
                        help="temperature of logits used for model input.")
    parser.add_argument("--output-lgt-temp", type=float, default=1,
                        help="temperature of logits used for model output.")
    parser.add_argument("--rl-output-lgt-temp", type=float, default=1,
                        help="temperature of logits used for model output.")
    parser.add_argument("--init-temp", type=float, default=0.1,
                        help="temperature of logits used in the initialization pass. High => uniform init.")
    parser.add_argument("--init-mode", type=str, default='original', choices=['random', 'original'])
    # lr
    parser.add_argument("--stepsize", type=float, default=0.1, help="learning rate in the backward pass.")
    parser.add_argument("--stepsize-ratio", type=float, default=1, help="")
    parser.add_argument("--stepsize-iters", type=int, default=1000, help="")
    # iterations
    parser.add_argument("--num-iters", type=int, default=1000)
    parser.add_argument("--min-iters", type=int, default=0, help="record best only after N iterations")
    parser.add_argument("--noise-iters", type=int, default=1, help="add noise at every N iterations")
    parser.add_argument("--optim-y-iters", type=int, default=5, help="optimize y at every N iterations")
    parser.add_argument("--paraphrase-iters", type=int, default=50, help="paraphrase at every N iterations")
    parser.add_argument("--win-anneal-iters", type=int, default=-1, help="froze the optimization window after N iters")
    parser.add_argument("--constraint-iters", type=int, default=1000,
                        help="add one more group of constraints from N iters")
    # gaussian noise
    parser.add_argument("--gs_mean", type=float, default=0.0)
    parser.add_argument("--gs_std", type=float, default=0.01)
    parser.add_argument("--large-noise-iters", type=str, default="-1", help="Example: '50,1000'")
    parser.add_argument("--large_gs_std", type=str, default="1", help="Example: '1,0.1'")

    args = parser.parse_args()
    return args

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    args = options()
    # device_ids = torch.cuda.device_count()
    generate_device_id = args.device_id
    guard_device_id = args.guard_device_id
    
    if args.pretrained_guard_model != "None":
        assert generate_device_id != guard_device_id, "generate_device_id and guard_device_id should be different, while we got generate_device_id = {}, guard_device_id = {}".format(generate_device_id, guard_device_id)
    else:
        guard_device_id = -1 # set -1 as unuse

    device = "cuda:{}".format(generate_device_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
    guard_model_device = "cuda:{}".format(guard_device_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
    
    # resolved through the HF cache (HF_HUB_CACHE)
    model_path_dicts = {"Llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
                        "Qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
                        "Qwen3.5-4b": "Qwen/Qwen3.5-4B",
                        "Qwen3.5-9b": "Qwen/Qwen3.5-9B",
                        }
    guard_model_path_dicts = {
            "d3-flash" : "vllm-sr/d3-flash",
            }
    model_path = model_path_dicts[args.pretrained_model]
    if args.pretrained_guard_model in guard_model_path_dicts:
        guard_model_path = guard_model_path_dicts[args.pretrained_guard_model]
    else:
        guard_model_path = ""
    if args.seed != -1:
        seed_everything(args.seed)
    # Load pretrained model
    if args.pretrained_model.startswith("Qwen3.5"):
        # bf16-native hybrid (Gated DeltaNet + attention) model; its nested config does not take use_cache here
        model, tokenizer = load_model_and_tokenizer(model_path,
                                                    low_cpu_mem_usage=True,
                                                    dtype=torch.bfloat16,
                                                    device=device)
    else:
        model, tokenizer = load_model_and_tokenizer(model_path,
                                                    low_cpu_mem_usage=True,
                                                    use_cache=False,
                                                    device=device)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    if guard_model_path != "":
        from d3_guard import D3Guard
        guard_model = D3Guard(guard_model_path, guard_model_device)
        guard_tokenizer = tokenizer  # candidate prompts are tokenised with the target tokenizer
        if guard_model.shares_vocab(tokenizer):
            # same token ids as d3: one-hot @ d3's own embedding table, exact
            guard_model_embedding_weights = guard_model.own_embeddings()
        else:
            guard_model_embedding_weights = guard_model.bridge_embeddings(
                tokenizer, model.get_input_embeddings().weight.shape[0])
        print("guard embeddings:", "d3 own table" if guard_model.shares_vocab(tokenizer) else "bridge",
              tuple(guard_model_embedding_weights.shape))
    elif args.pretrained_guard_model != "None":
        guard_model = DNNClassifier()
        guard_model.load_state_dict(torch.load(f"../proxy_model/model/{args.pretrained_guard_model}.pth"))
        guard_model.to(guard_device_id)
        guard_model.eval()
        guard_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            trust_remote_code=True,
            use_fast=False,
            use_cache=True,
        )
        if not guard_tokenizer.pad_token:
            guard_tokenizer.pad_token = guard_tokenizer.eos_token
        guard_model_embedding_weights = None
    else:
        guard_model, guard_tokenizer, guard_model_embedding_weights = None, None, None

    # Freeze GPT-2 weights
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    if "suffix" in args.mode:
        from attack_suffix import attack_generation
        print("suffix")
        attack_generation(model, tokenizer, device, args, guard_model = guard_model, guard_tokenizer = guard_tokenizer, guard_model_embedding_weights = guard_model_embedding_weights, guard_model_device = guard_model_device)
    elif "paraphrase" in args.mode:
        print("paraphrase")
        from attack_paraphrase import attack_generation
        attack_generation(model, tokenizer, device, args, guard_model = guard_model, guard_tokenizer = guard_tokenizer, guard_model_embedding_weights = guard_model_embedding_weights, guard_model_device = guard_model_device)
    elif "control" in args.mode:
        print("control")
        from attack_control import attack_generation
        attack_generation(model, tokenizer, device, args, guard_model = guard_model, guard_tokenizer = guard_tokenizer, guard_model_embedding_weights = guard_model_embedding_weights, guard_model_device = guard_model_device)

if __name__ == "__main__":
    main()

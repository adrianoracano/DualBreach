# Target-driven Initialization (TDI): scenario-based persuasion.
# Every harmful query is rewritten into a situational scenario that keeps its intent, using the local GGUF paraphraser of
# MTO/local_paraphraser.py (the paper's QwenPlus/QwenMax/GPT-4o/GPT-4 API models are not needed).
#
# usage (from TDI/, CPU only: run it through Slurm, e.g. sbatch --cpus-per-task=32 --wrap "python scenario_initialize.py advbench"):
#   python scenario_initialize.py {advbench,DNA,harmBench} [--start N] [--end M]
# reads ../data/<name>_100.csv and writes ../data/<name>_scenario_100.csv (the file MTO's `<name>_scenario_100` dataset loads)
# with an extra `prompt` column; rows that already have a prompt are skipped, so an interrupted run can be resumed.
import argparse
import os
import sys

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MTO"))
from local_paraphraser import harmful_paraphrase  # noqa: E402

# MTO's attack_suffix.py loads the advbench scenarios from a file with this suffix
OUTPUT_NAME = {"advbench": "advbench_scenario_100_update.csv"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["advbench", "DNA", "harmBench"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None, help="exclusive")
    args = parser.parse_args()

    out_path = os.path.join(DATA_DIR, OUTPUT_NAME.get(args.dataset, f"{args.dataset}_scenario_100.csv"))
    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
    else:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{args.dataset}_100.csv"))
        df["prompt"] = None
    df["prompt"] = df["prompt"].astype(object)

    for idx in range(args.start, min(args.end if args.end is not None else len(df), len(df))):
        if isinstance(df.loc[idx, "prompt"], str) and df.loc[idx, "prompt"].strip():
            continue
        original_prompt = df.loc[idx, "harmful"]
        print(f"-----------start----------- index {idx}\nOriginal prompt: {original_prompt}", flush=True)
        try:
            df.loc[idx, "prompt"] = harmful_paraphrase(original_prompt)
        except RuntimeError as e:  # no valid rewrite after all retries: leave the row empty and move on
            print(f"index {idx} skipped: {e}", flush=True)
            continue
        print(f"Jailbreak prompt: {df.loc[idx, 'prompt']}\n-----------end-----------", flush=True)
        # saved after every row: each paraphrase takes minutes on CPU
        df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()

"""Print the records of a MTO result .jsonl in a readable form.

usage: python show_results.py <results.jsonl | run_tag> [--id N ...] [--rounds] [--full] [--width W]

`run_tag` is the name given to `slurm_submit.sh` / `--run_tag` (e.g. v1_q4b_ids_s5to9); the file is then looked up in
outputs/GuardBreaker/suffix/attack_evaluate/*/. Per sample it shows the query, the best guard candidate, the model
response (only generated for candidates that passed the guard, so it is empty when the guard was never fooled), the
judge/guard counts and a summary of the guard trace. --rounds adds one line per guard round, --full disables truncation.
"""
import argparse
import glob
import json
import statistics as st
import textwrap

ROOT = "outputs/GuardBreaker/suffix/attack_evaluate/"


def find(arg):
    if arg.endswith(".jsonl"):
        return [arg]
    files = sorted(glob.glob(ROOT + f"*/*_v2_{arg}.jsonl"))
    if not files:
        raise SystemExit(f"no result file for tag '{arg}' under {ROOT}")
    return files


def block(text, width, full, limit=1200):
    text = (text or "").strip() or "(empty)"
    if not full and len(text) > limit:
        text = text[:limit] + f" ... [{len(text) - limit} more chars]"
    return "\n".join(textwrap.fill(p, width, initial_indent="    ", subsequent_indent="    ") if p.strip() else ""
                     for p in text.splitlines())


def show(r, args):
    tr = r.get("guard_trace") or []
    bar = "=" * args.width
    print(bar)
    print(f"sample {r['harmful_id']}  result: {r['result'].upper()}  | {r.get('pretrained_model')}  guard_loss={r.get('guard_loss')}  "
          f"guard_align={r.get('guard_align')}  iters={r['iter_amount']}  paraphrases={r['paraphrase_amount']}  time={r['time']:.0f}s")
    print(bar)
    print("QUERY:\n" + block(r["harmful"], args.width, True))
    print("TARGET STRING:\n" + block(r["target"], args.width, True))

    n = r["total_query"]
    print(f"\nGUARD: {n - r['fail_guard']}/{n} candidates passed (P(safe) > 0.5), "
          f"{r['fail_judge']} of the {r['total_judge']} judged responses failed the LLM judge; "
          f"jailbreak_score={r['jailbreak_score']}  keyword_check={r['final_keyword']}")
    print(f"final candidate: P(safe)={r['final_safe_score']:.4g}  P(unsafe)={r['final_unsafe_score']:.4g}")

    if tr:
        ps = [t["p_safe"] for t in tr]
        mg = [t["margin"] for t in tr]
        b = min(tr, key=lambda t: t["margin"])
        half = len(tr) // 2
        print(f"guard trace ({len(tr)} rounds): best P(safe)={max(ps):.4g} (margin {min(mg):.2f}, iter {b['iter']}), "
              f"median P(safe)={st.median(ps):.3g}, mean margin first/second half={st.mean(mg[:half]):.2f}/{st.mean(mg[half:]):.2f}")
        al = [t["align_frac"] for t in tr if t.get("align_frac") is not None]
        zero = sum((t["step_nnz"] or 0) == 0 for t in tr)
        extra = f"; mean align_frac={st.mean(al):.2f}" if al else ""
        print(f"  guard step: median |step|_1={st.median(t['step_l1'] for t in tr):.3g}, median grad_norm={st.median(t['grad_norm'] for t in tr):.3g}, "
              f"zero-step rounds={zero}{extra}")
        if "sm_maxp_mean" in tr[-1]:
            print(f"  softmax (last round): mean top-1 prob={tr[-1]['sm_maxp_mean']:.3f}, mean top1-top2 gap={tr[-1]['gap_mean']:.2f}, "
                  f"entries with p>1e-10={tr[-1]['sm_active']:,}")

    advs = r.get("prompt_with_advs") or []
    if tr and advs:
        i = min(range(len(tr)), key=lambda k: tr[k]["margin"])
        print(f"\nBEST CANDIDATE by guard margin (round {i + 1}, iter {tr[i]['iter']}, P(safe)={tr[i]['p_safe']:.4g}):")
        print(block(advs[i], args.width, args.full))
    if r.get("prompt_with_adv"):
        print("\nPROMPT SENT TO THE MODEL (candidate that passed the guard):")
        print(block(r["prompt_with_adv"], args.width, args.full))
    print("\nMODEL RESPONSE:")
    print(block(r.get("final_response"), args.width, args.full))

    js = r.get("jailbreak_scores") or []
    if any(js):
        print(f"\njudge scores per round (non-zero): { {k + 1: s for k, s in enumerate(js) if s} }")

    if args.rounds and tr:
        print("\nper-round trace:")
        print(f"  {'rnd':>3} {'iter':>4} {'P(safe)':>9} {'margin':>7} {'|step|_1':>9} {'grad':>7} {'align':>5} {'judge':>5}  candidate suffix")
        for k, t in enumerate(tr):
            cand = advs[k][len(r["harmful"]):].strip().replace("\n", " ") if k < len(advs) else ""
            print(f"  {k + 1:>3} {t['iter']:>4} {t['p_safe']:>9.3g} {t['margin']:>7.2f} {t['step_l1']:>9.3g} {t['grad_norm']:>7.3g} "
                  f"{(t.get('align_frac') if t.get('align_frac') is not None else float('nan')):>5.2f} {js[k] if k < len(js) else '':>5}  {cand[:args.width - 50]}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="+", help="results .jsonl path(s) or run tag(s)")
    ap.add_argument("--id", type=int, nargs="*", help="only these harmful_ids")
    ap.add_argument("--rounds", action="store_true", help="print one line per guard round")
    ap.add_argument("--full", action="store_true", help="do not truncate long texts")
    ap.add_argument("--width", type=int, default=110)
    args = ap.parse_args()

    recs = []
    for a in args.run:
        for f in find(a):
            recs += [json.loads(l) for l in open(f) if l.strip()]
    recs = [r for r in recs if not args.id or r["harmful_id"] in args.id]
    for r in sorted(recs, key=lambda r: r["harmful_id"]):
        show(r, args)
    if recs:
        ok = sum(r["result"] == "success" for r in recs)
        passed = sum(r["total_query"] - r["fail_guard"] > 0 for r in recs)
        print(f"{len(recs)} samples: {ok} successful attacks, {passed} with at least one candidate passing the guard")


if __name__ == "__main__":
    main()

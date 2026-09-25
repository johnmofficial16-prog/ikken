"""Aggregate results/*.json into results/summary.json and print markdown tables for RESULTS.md."""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def load(pattern):
    out = {}
    for p in sorted(glob.glob(os.path.join(RES, pattern))):
        with open(p) as fh:
            out[os.path.basename(p)] = json.load(fh)
    return out


def fmt(x, nd=0):
    return "-" if x is None else ("{:,.%df}" % nd).format(x)


def exactness_table(ex):
    lines = ["| backbone | attn | dtype | S | K | max abs diff, question-block hidden (max value) | max abs diff, logits | max abs diff, probs | argmax agree | state vs state-only | reversed order (probs) | padded batch (logits) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ctrl = []
    for f, d in ex.items():
        for res in d["results"]:
            for r in res["runs"]:
                p = r["packed_vs_single"]
                lines.append("| %s | %s | %s | %d | %d | %.1e (%.0f) | %.1e | %.1e | %s | %.1e | %.1e | %.1e |" % (
                    res["model"], res["attn"], res["dtype"].replace("torch.", ""), r["S"], r["K"],
                    p["max_abs_diff_block_hidden"], p["max_abs_hidden_value"], p["max_abs_diff_logits"],
                    p["max_abs_diff_probs"], p["argmax_agree"], r["state_hidden_max_abs_diff_vs_state_only"],
                    r["reverse_order_max_abs_diff_probs"], r["padded_batch_max_abs_diff_logits"]))
                if "negative_controls" in r:
                    for k, c in r["negative_controls"].items():
                        ctrl.append("| %s | %s | %s | %d | %s | %.2g | %.2g | %s |" % (
                            res["model"], res["attn"], res["dtype"].replace("torch.", ""), r["S"], k,
                            c["max_abs_diff_block_hidden"], c["max_abs_diff_logits"], c["argmax_agree"]))
    lines += ["", "| backbone | attn | dtype | S | negative control | max abs diff, block hidden | max abs diff, logits | argmax agree |",
              "|---|---|---|---|---|---|---|---|"] + ctrl
    return "\n".join(lines)


def latency_table(lat):
    lines = []
    for f, d in lat.items():
        lines.append("\n**%s** (%s; battery %s; %s)\n" % (
            d["model"], f, d["env"].get("battery"), d["env"].get("power_scheme", "").split("(")[-1].rstrip(")")))
        lines.append("| S | K | Q tokens | baseline: K rows x len (bs) | baseline median / p90 ms (n) | read-once packed median / p90 ms (n) | speedup | read-once prefix-KV median / p90 ms (n) | speedup | best read-once cost / 1 question |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        one = {}
        for r in d["runs"]:
            if r["K"] == 1 and "median_ms" in r["baseline"]:
                one[r["S"]] = r["baseline"]["median_ms"]
        for r in d["runs"]:
            ro, bl, kv = r["readonce"], r["baseline"], r.get("prefixkv")
            if "median_ms" in bl:
                bls = "%s / %s (%d)%s" % (fmt(bl["median_ms"]), fmt(bl["p90_ms"]), bl["n"],
                                          " single rep" if bl.get("single_rep_no_warmup") else "")
                sp = "%.1fx" % r["speedup_median"]
                spk = "%.1fx" % r["speedup_prefixkv_median"] if kv else "-"
            else:
                bls = "not run; est. %s (ESTIMATE)" % fmt(bl["estimated_ms_per_rep_ESTIMATE"])
                sp = "~%.0fx (ESTIMATE)" % r["speedup_ESTIMATE"]
                spk = "~%.0fx (ESTIMATE)" % r["speedup_prefixkv_ESTIMATE"] if kv else "-"
            best = min(ro["median_ms"], kv["median_ms"]) if kv else ro["median_ms"]
            rel = ("%.2fx" % (best / one[r["S"]])) if r["S"] in one else "-"
            kvs = "%s / %s (%d)" % (fmt(kv["median_ms"]), fmt(kv["p90_ms"]), kv["n"]) if kv else "-"
            lines.append("| %d | %d | %d | %d x %d (%d) | %s | %s / %s (%d) | %s | %s | %s | %s |" % (
                r["S"], r["K"], r["question_tokens"], r["baseline_rows"], r["baseline_row_len_max"],
                r["baseline_batch_size"], bls, fmt(ro["median_ms"]), fmt(ro["p90_ms"]), ro["n"], sp, kvs, spk, rel))
    return "\n".join(lines)


def train_table(tr):
    lines = ["| run | arch | backbone | style | steps | median step s | train s | acc | chance | ECE | NLL | Brier | lookup acc | numeric acc |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f, d in tr.items():
        a, ev = d["args"], d["eval"]
        o = ev["overall"]
        lines.append("| %s | %s | %s | %s | %d | %.2f | %.0f | %.3f | %.3f | %.3f | %.3f | %.3f | %s | %s |" % (
            f, a["arch"], a["model"], a["style"], a["steps"], d["median_step_s"], d["train_seconds"],
            o["accuracy"], o["chance"], o["ece"], o["nll"], o["brier"],
            ev.get("family:lookup", {}).get("accuracy", "-"), ev.get("family:numeric", {}).get("accuracy", "-")))
    return "\n".join(lines)


def template_table(tr):
    runs = list(tr.items())
    keys = sorted({k for _, d in runs for k in d["eval"] if k.startswith(("template:", "options:"))})
    lines = ["| slice | n | chance | " + " | ".join("%s acc / ECE" % f.replace("train_", "").replace(".json", "") for f, _ in runs) + " |",
             "|---|---|---|" + "---|" * len(runs)]
    for k in keys:
        first = next(d["eval"][k] for _, d in runs if k in d["eval"])
        cells = []
        for _, d in runs:
            m = d["eval"].get(k)
            cells.append("-" if not m else "%.3f / %.3f" % (m["accuracy"], m["ece"]))
        lines.append("| %s | %d | %.3f | %s |" % (k, first["n"], first["chance"], " | ".join(cells)))
    return "\n".join(lines)


if __name__ == "__main__":
    ex, lat, tr = load("exactness_*.json"), load("latency_*.json"), load("train_*.json")
    summary = {"exactness_files": list(ex), "latency_files": list(lat), "train_files": list(tr)}
    print("## Exactness\n")
    print(exactness_table(ex))
    print("\n## Latency\n")
    print(latency_table(lat))
    if tr:
        print("\n## Training\n")
        print(train_table(tr))
        print()
        print(template_table(tr))
    with open(os.path.join(RES, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

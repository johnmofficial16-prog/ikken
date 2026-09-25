"""Build smoke.ipynb: a self-contained Kaggle notebook that writes readonce.py, synth.py and smoke.py
into the working directory, installs the pinned packages and runs the smoke test.

Usage: python build_notebook.py                     (writes smoke.ipynb next to this file)
       python build_notebook.py --throughput-reps 5 (writes throughput.ipynb: only the repeated
                                                     read-once vs per-question throughput pair)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.join(HERE, "..", "01-read-once-prototype")


def src(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def code(text):
    lines = text.splitlines(keepends=True)
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


summary = r'''import json
r = json.load(open("/kaggle/working/smoke_results.json"))
for name, t in r["tests"].items():
    if not t.get("ok"):
        print(f"ERR  {name}: {t.get('error') or t.get('skipped')}")
        continue
    res = t["result"]
    if name.startswith("train_"):
        keep = ("oom", "median_step_ms", "tokens_per_s", "questions_per_s", "peak_mem_gb", "loss_first", "loss_last", "nonfinite_losses", "grad_scaler_scale_end")
    elif name.startswith("activations_"):
        res = {k: {kk: v[kk] for kk in ("global_max_abs", "headroom_vs_fp16_max", "nonfinite_values", "logits_finite")} for k, v in res.items() if k in ("fp32", "fp16")}
        keep = None
    elif name == "env":
        keep = ("torch", "cuda", "transformers", "gpus")
    else:
        keep = None
    if keep:
        res = {k: res[k] for k in keep if k in res}
    print(f"OK   {name} ({t['seconds']} s): {json.dumps(res)[:700]}")
'''

REPS = int(sys.argv[sys.argv.index("--throughput-reps") + 1]) if "--throughput-reps" in sys.argv else 0

cells = [
    md("# Read-once encoder: T4 smoke test\n\n"
       "Settings: **Accelerator = GPU T4 x2**, **Internet = on**. Then *Run all*.\n\n"
       "Writes `/kaggle/working/smoke_results.json`. Each test is independent; results are saved after every test."),
    code("!nvidia-smi\n"
         "%pip install -q transformers==5.17.0 tokenizers==0.23.2 huggingface_hub==1.32.0 safetensors==0.8.0"),
    code("%%writefile readonce.py\n" + src(os.path.join(PROTO, "readonce.py"))),
    code("%%writefile synth.py\n" + src(os.path.join(PROTO, "synth.py"))),
    code("%%writefile smoke.py\n" + src(os.path.join(HERE, "smoke.py"))),
    code("!python smoke.py --budget-min 50" + (f" --throughput-reps {REPS}" if REPS else "")),
    code(summary),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "throughput.ipynb" if REPS else "smoke.ipynb")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)
print(out, os.path.getsize(out), "bytes")

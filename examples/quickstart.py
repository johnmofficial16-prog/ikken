"""Quickstart: encode one state once, then read the option markers of several questions.

Uses Ettin-17m (MIT licence) from the Hugging Face Hub, pinned to the safetensors conversion of the
checkpoint, because the main branch ships a pickle file only.

A freshly loaded encoder has no trained decision head. This example shows the mechanics and the
parity check, not answers.

Usage: python examples/quickstart.py
"""
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ikken import ReadOnceEncoder, choice_block

REPO, REVISION = "jhu-clsp/ettin-encoder-17m", "59c53d9a19ee484b200676d55e4ae240528b2bb9"

tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
# Load through the masked-LM class: these safetensors store the tied input embedding as decoder.weight,
# so a bare AutoModel load would leave the embeddings randomly initialised.
mlm = AutoModelForMaskedLM.from_pretrained(REPO, revision=REVISION, use_safetensors=True,
                                           attn_implementation="sdpa")
encoder = mlm.model.eval()

ro = ReadOnceEncoder(encoder, tokenizer)
state = ("Ticket 4812 from Northwind (Enterprise plan), opened 2 days ago, last reply 26 hours ago: "
         "'Our dashboard has shown no data since Monday's upgrade and 40 users are blocked.'")
questions = [
    choice_block(tokenizer, "Priority?", ["P1", "P2", "P3", "P4"]),
    choice_block(tokenizer, "Is the customer blocked?", ["yes", "no"]),
    choice_block(tokenizer, "Which team should take it?", ["billing", "platform", "mobile", "integrations"]),
]

batch, rows = ro.pack([state], [questions])
with torch.no_grad():
    hidden = ro.encode(batch)
for name, markers in zip(["priority", "blocked", "team"], ro.markers(hidden, rows)[0]):
    print(f"{name:>8}: {markers.shape[0]} option markers x {markers.shape[1]} dims")

print("parity (packed vs each question alone):", ro.check_parity(state, questions))

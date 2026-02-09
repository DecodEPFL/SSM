"""Minimal LRA ListOps benchmark runner for SSM models.

Example:
    python scripts/run_lra_listops.py --train-steps 200 --eval-steps 50

Requires:
    pip install datasets
"""

from __future__ import annotations

import argparse
import math
from itertools import islice

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_ssm import DeepSSM


class SSMListOpsClassifier(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_classes: int,
        d_model: int,
        d_state: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.ssm = DeepSSM(
            d_input=d_model,
            d_output=d_model,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.ssm(x)
        mask = attention_mask.unsqueeze(-1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(x)


def build_collate_fn(max_length: int, pad_token_id: int) -> callable:
    def collate(batch: list[dict[str, object]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequences = [torch.tensor(example["input_ids"][:max_length], dtype=torch.long) for example in batch]
        labels = torch.tensor([example["label"] for example in batch], dtype=torch.long)

        lengths = [seq.numel() for seq in sequences]
        max_len = max(lengths)
        padded = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(sequences), max_len), dtype=torch.float32)

        for idx, seq in enumerate(sequences):
            padded[idx, : seq.numel()] = seq
            attention[idx, : seq.numel()] = 1.0

        return padded, attention, labels

    return collate


def infer_vocab_size(dataset, max_scan: int) -> int:
    feature = dataset.features.get("input_ids")
    if feature is not None and hasattr(feature, "feature"):
        inner = feature.feature
        if getattr(inner, "num_classes", None):
            return inner.num_classes

    max_token = 0
    for example in islice(dataset, max_scan):
        max_token = max(max_token, max(example["input_ids"]))
    return max_token + 1


def infer_num_classes(dataset) -> int:
    feature = dataset.features.get("label")
    if feature is not None and getattr(feature, "num_classes", None):
        return feature.num_classes
    labels = dataset["label"]
    return len(set(labels))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal LRA ListOps benchmark with DeepSSM.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-vocab-scan", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dataset = load_dataset("lra", "listops")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    vocab_size = infer_vocab_size(train_dataset, args.max_vocab_scan)
    num_classes = infer_num_classes(train_dataset)

    model = SSMListOpsClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(args.device)

    collate_fn = build_collate_fn(args.max_length, pad_token_id=0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    train_iter = iter(train_loader)
    for step in tqdm(range(1, args.train_steps + 1), desc="train", unit="step"):
        try:
            input_ids, attention_mask, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, attention_mask, labels = next(train_iter)

        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        labels = labels.to(args.device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % max(1, math.ceil(args.train_steps / 10)) == 0:
            tqdm.write(f"step {step}: loss={loss.item():.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for step, (input_ids, attention_mask, labels) in enumerate(eval_loader, start=1):
            if step > args.eval_steps:
                break
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            labels = labels.to(args.device)

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    accuracy = correct / total if total else 0.0
    print(f"eval accuracy (first {total} examples): {accuracy:.4f}")


if __name__ == "__main__":
    main()

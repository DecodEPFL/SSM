"""Minimal LRA benchmark runner for SSM models (single- or pair-sequence tasks).

Examples:
    python scripts/run_lra.py --dataset-config listops
    python scripts/run_lra.py --dataset-config pathfinder
    python scripts/run_lra.py --dataset-config retrieval --pair-mode

Requires:
    pip install datasets
"""

from __future__ import annotations

import argparse
import math
from itertools import islice
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from datasets import load_dataset  # type: ignore
    from datasets.exceptions import DatasetNotFoundError  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "The 'datasets' package is required. Install it with: pip install datasets"
    ) from exc

from neural_ssm import DeepSSM


def infer_vocab_size(dataset, fields: Iterable[str], max_scan: int) -> int:
    # Try dataset features first
    for field in fields:
        feature = dataset.features.get(field)
        if feature is not None and hasattr(feature, "feature"):
            inner = feature.feature
            if getattr(inner, "num_classes", None):
                return inner.num_classes

    # Fallback: scan examples
    max_token = 0
    for example in islice(dataset, max_scan):
        for field in fields:
            if field in example:
                seq = example[field]
                if len(seq) == 0:
                    continue
                max_token = max(max_token, max(seq))
    return max_token + 1


def infer_num_classes(dataset, label_field: str) -> int:
    feature = dataset.features.get(label_field)
    if feature is not None and getattr(feature, "num_classes", None):
        return feature.num_classes
    labels = dataset[label_field]
    return len(set(labels))


def pad_sequences(seqs: list[torch.Tensor], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = [seq.numel() for seq in seqs]
    max_len = max(lengths)
    padded = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(seqs), max_len), dtype=torch.float32)
    for idx, seq in enumerate(seqs):
        padded[idx, : seq.numel()] = seq
        attention[idx, : seq.numel()] = 1.0
    return padded, attention


def build_single_collate(max_length: int, pad_token_id: int, label_field: str):
    def collate(batch: list[dict[str, object]]):
        sequences = [
            torch.tensor(example["input_ids"][:max_length], dtype=torch.long)
            for example in batch
        ]
        labels = torch.tensor([example[label_field] for example in batch], dtype=torch.long)
        padded, attention = pad_sequences(sequences, pad_token_id)
        return padded, attention, labels

    return collate


def build_pair_collate(max_length: int, pad_token_id: int, field_a: str, field_b: str, label_field: str):
    def collate(batch: list[dict[str, object]]):
        seqs_a = [
            torch.tensor(example[field_a][:max_length], dtype=torch.long)
            for example in batch
        ]
        seqs_b = [
            torch.tensor(example[field_b][:max_length], dtype=torch.long)
            for example in batch
        ]
        labels = torch.tensor([example[label_field] for example in batch], dtype=torch.long)
        padded_a, attention_a = pad_sequences(seqs_a, pad_token_id)
        padded_b, attention_b = pad_sequences(seqs_b, pad_token_id)
        return padded_a, attention_a, padded_b, attention_b, labels

    return collate


def build_local_collate(max_length: int, pad_token_id: int):
    def collate(batch: list[tuple[torch.Tensor, int]]):
        sequences = [seq[:max_length] for seq, _ in batch]
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        padded, attention = pad_sequences(sequences, pad_token_id)
        return padded, attention, labels

    return collate


def mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def last_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.sum(dim=1).long().clamp(min=1)
    idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1)
    idx = idx.expand(-1, 1, x.size(-1))
    return x.gather(1, idx).squeeze(1)


def pool_sequence(x: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mean":
        return mean_pool(x, mask)
    if mode == "last":
        return last_pool(x, mask)
    raise ValueError(f"Unknown pooling mode: {mode}")


class SSMSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_classes: int,
        d_model: int,
        d_state: int,
        n_layers: int,
        dropout: float,
        param: str,
        ff: str,
        gamma: float | None,
        train_gamma: bool,
        mode: str,
        pooling: str,
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
            param=param,
            ff=ff,
            gamma=gamma,
            train_gamma=train_gamma,
        )
        self.classifier = nn.Linear(d_model, num_classes)
        self.mode = mode
        self.pooling = pooling

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x, _ = self.ssm(x, mode=self.mode)
        pooled = pool_sequence(x, attention_mask, self.pooling)
        return self.classifier(pooled)


class SSMPairClassifier(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_classes: int,
        d_model: int,
        d_state: int,
        n_layers: int,
        dropout: float,
        param: str,
        ff: str,
        gamma: float | None,
        train_gamma: bool,
        mode: str,
        pooling: str,
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
            param=param,
            ff=ff,
            gamma=gamma,
            train_gamma=train_gamma,
        )
        self.classifier = nn.Linear(d_model * 4, num_classes)
        self.mode = mode
        self.pooling = pooling

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x, _ = self.ssm(x, mode=self.mode)
        return pool_sequence(x, attention_mask, self.pooling)

    def forward(
        self,
        input_ids_a: torch.Tensor,
        attention_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_b: torch.Tensor,
    ) -> torch.Tensor:
        a = self.encode(input_ids_a, attention_a)
        b = self.encode(input_ids_b, attention_b)
        features = torch.cat([a, b, (a - b).abs(), a * b], dim=-1)
        return self.classifier(features)


def resolve_pair_fields(example: dict[str, object]) -> tuple[str, str] | None:
    candidates = [
        ("input_ids1", "input_ids2"),
        ("input_ids_1", "input_ids_2"),
        ("input_ids_a", "input_ids_b"),
    ]
    for a, b in candidates:
        if a in example and b in example:
            return a, b
    return None


def get_column_names(dataset) -> list[str]:
    if hasattr(dataset, "column_names") and dataset.column_names:
        return list(dataset.column_names)
    if hasattr(dataset, "features") and dataset.features:
        return list(dataset.features.keys())
    try:
        first = next(iter(dataset))
        if isinstance(first, dict):
            return list(first.keys())
    except Exception:
        pass
    return []


def resolve_label_field(dataset, preferred: str | None) -> str:
    columns = get_column_names(dataset)
    if preferred:
        if preferred in columns:
            return preferred
        raise ValueError(
            f"Label field '{preferred}' not found. Columns: {columns}"
        )

    candidates = ["label", "labels", "target", "targets", "y"]
    for name in candidates:
        if name in columns:
            return name

    raise ValueError(
        "No label field found. Specify --label-field. "
        f"Columns: {columns}"
    )


class ListOpsDataset(torch.utils.data.Dataset):
    def __init__(self, sequences: list[torch.Tensor], labels: list[int]) -> None:
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.sequences[idx], self.labels[idx]


def tokenize_listops_sequence(text: str, vocab: dict[str, int], build_vocab: bool) -> list[int]:
    tokens = text.strip().split()
    ids = []
    for tok in tokens:
        if tok in vocab:
            ids.append(vocab[tok])
        else:
            if build_vocab:
                vocab[tok] = len(vocab)
                ids.append(vocab[tok])
            else:
                ids.append(vocab.get("<unk>", 1))
    return ids


def load_listops_tsv(
    path: Path,
    vocab: dict[str, int] | None,
    build_vocab: bool,
) -> tuple[list[torch.Tensor], list[int], dict[str, int]]:
    sequences: list[torch.Tensor] = []
    labels: list[int] = []
    if vocab is None:
        vocab = {"<pad>": 0, "<unk>": 1}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                continue

            # Skip header lines if present
            if fields[0].lower() in {"label", "labels"} or fields[-1].lower() in {"label", "labels"}:
                continue

            label = None
            seq = None
            if fields[0].lstrip("-").isdigit():
                label = int(fields[0])
                seq = fields[1]
            elif fields[-1].lstrip("-").isdigit():
                label = int(fields[-1])
                seq = fields[0]
            else:
                continue

            token_ids = tokenize_listops_sequence(seq, vocab, build_vocab)
            if not token_ids:
                continue
            sequences.append(torch.tensor(token_ids, dtype=torch.long))
            labels.append(label)

    return sequences, labels, vocab


def resolve_listops_paths(
    listops_dir: Path,
    train_file: str | None,
    val_file: str | None,
    test_file: str | None,
) -> tuple[Path, Path | None, Path | None]:
    if train_file:
        train_path = listops_dir / train_file
        val_path = listops_dir / val_file if val_file else None
        test_path = listops_dir / test_file if test_file else None
        return train_path, val_path, test_path

    files = [p for p in listops_dir.iterdir() if p.is_file() and p.suffix.lower() == ".tsv"]
    if not files:
        raise FileNotFoundError(f"No .tsv files found in {listops_dir}")

    def pick(keys: list[str]) -> Path | None:
        hits = [p for p in files if any(k in p.name.lower() for k in keys)]
        if not hits:
            return None
        hits.sort(key=lambda p: (0 if "basic" in p.name.lower() else 1, len(p.name)))
        return hits[0]

    train_path = pick(["train"])
    val_path = pick(["val", "valid", "dev"])
    test_path = pick(["test"])

    if train_path is None:
        raise FileNotFoundError(
            "Could not find a train split. Provide --listops-train explicitly."
        )

    return train_path, val_path, test_path


def safe_load_dataset(name: str, config: str | None):
    # Try (name, config) first, then fall back to just (name).
    try:
        if config:
            return load_dataset(name, config)
        return load_dataset(name)
    except (DatasetNotFoundError, ValueError) as exc:
        if config:
            try:
                return load_dataset(name)
            except Exception:  # pragma: no cover - surface original error
                pass
        msg = (
            f"Could not load dataset '{name}' with config '{config}'.\n"
            "Tips:\n"
            "- The HF dataset 'lra-benchmarks' is no longer available.\n"
            "- Try '--dataset-name OpenNLPLab/lra' (pair-mode retrieval).\n"
            "- For the official LRA tasks (ListOps/Text/Pathfinder), "
            "download the LRA release data and adapt the loader."
        )
        raise RuntimeError(msg) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LRA benchmarks with DeepSSM.")
    parser.add_argument("--dataset-name", type=str, default="local-listops")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--listops-dir", type=str, default=None)
    parser.add_argument("--listops-train", type=str, default=None)
    parser.add_argument("--listops-val", type=str, default=None)
    parser.add_argument("--listops-test", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ff", type=str, default="LGLU")
    parser.add_argument("--param", type=str, default="lru")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--train-gamma", action="store_true")
    parser.add_argument("--mode", type=str, default="scan")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "last"])
    parser.add_argument("--max-vocab-scan", type=int, default=50000)
    parser.add_argument("--label-field", type=str, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pair-mode", action="store_true", help="Force pair-sequence mode.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.dataset_name == "local-listops" or args.listops_dir is not None:
        if args.listops_dir is None:
            raise ValueError("For local ListOps, pass --listops-dir /path/to/listops")

        listops_dir = Path(args.listops_dir).expanduser()
        train_path, val_path, test_path = resolve_listops_paths(
            listops_dir,
            args.listops_train,
            args.listops_val,
            args.listops_test,
        )

        train_seqs, train_labels, vocab = load_listops_tsv(train_path, None, build_vocab=True)
        val_seqs, val_labels, vocab = ([], [], vocab)
        test_seqs, test_labels, vocab = ([], [], vocab)

        if val_path is not None and val_path.exists():
            val_seqs, val_labels, vocab = load_listops_tsv(val_path, vocab, build_vocab=False)
        if test_path is not None and test_path.exists():
            test_seqs, test_labels, vocab = load_listops_tsv(test_path, vocab, build_vocab=False)

        train_dataset = ListOpsDataset(train_seqs, train_labels)
        eval_dataset = ListOpsDataset(val_seqs, val_labels) if val_seqs else ListOpsDataset(test_seqs, test_labels)

        vocab_size = len(vocab)
        num_classes = max(train_labels) + 1 if train_labels else 1
        is_pair = False
        field_a, field_b = "", ""
        collate_fn = build_local_collate(args.max_length, pad_token_id=0)
    else:
        dataset = safe_load_dataset(args.dataset_name, args.dataset_config)
        train_dataset = dataset["train"]
        eval_dataset = dataset.get("validation") or dataset.get("test")

        first_example = train_dataset[0]
        pair_fields = resolve_pair_fields(first_example)
        is_pair = args.pair_mode or pair_fields is not None

        if is_pair:
            if pair_fields is None:
                raise ValueError("Pair mode requested but no pair fields found in dataset.")
            field_a, field_b = pair_fields
            vocab_size = infer_vocab_size(train_dataset, [field_a, field_b], args.max_vocab_scan)
        else:
            field_a, field_b = "input_ids", ""
            vocab_size = infer_vocab_size(train_dataset, ["input_ids"], args.max_vocab_scan)

        label_field = resolve_label_field(train_dataset, args.label_field)
        num_classes = infer_num_classes(train_dataset, label_field)

    if args.dataset_name == "local-listops" or args.listops_dir is not None:
        model = SSMSequenceClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            d_model=args.d_model,
            d_state=args.d_state,
            n_layers=args.n_layers,
            dropout=args.dropout,
            param=args.param,
            ff=args.ff,
            gamma=args.gamma,
            train_gamma=args.train_gamma,
            mode=args.mode,
            pooling=args.pooling,
        ).to(args.device)
    elif is_pair:
        model = SSMPairClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            d_model=args.d_model,
            d_state=args.d_state,
            n_layers=args.n_layers,
            dropout=args.dropout,
            param=args.param,
            ff=args.ff,
            gamma=args.gamma,
            train_gamma=args.train_gamma,
            mode=args.mode,
            pooling=args.pooling,
        ).to(args.device)
        collate_fn = build_pair_collate(
            args.max_length,
            pad_token_id=0,
            field_a=field_a,
            field_b=field_b,
            label_field=label_field,
        )
    else:
        model = SSMSequenceClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            d_model=args.d_model,
            d_state=args.d_state,
            n_layers=args.n_layers,
            dropout=args.dropout,
            param=args.param,
            ff=args.ff,
            gamma=args.gamma,
            train_gamma=args.train_gamma,
            mode=args.mode,
            pooling=args.pooling,
        ).to(args.device)
        collate_fn = build_single_collate(args.max_length, pad_token_id=0, label_field=label_field)

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
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        if is_pair:
            input_ids_a, attention_a, input_ids_b, attention_b, labels = batch
            input_ids_a = input_ids_a.to(args.device)
            attention_a = attention_a.to(args.device)
            input_ids_b = input_ids_b.to(args.device)
            attention_b = attention_b.to(args.device)
            labels = labels.to(args.device)
            logits = model(input_ids_a, attention_a, input_ids_b, attention_b)
        else:
            input_ids, attention_mask, labels = batch
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
        for step, batch in enumerate(eval_loader, start=1):
            if step > args.eval_steps:
                break

            if is_pair:
                input_ids_a, attention_a, input_ids_b, attention_b, labels = batch
                input_ids_a = input_ids_a.to(args.device)
                attention_a = attention_a.to(args.device)
                input_ids_b = input_ids_b.to(args.device)
                attention_b = attention_b.to(args.device)
                labels = labels.to(args.device)
                logits = model(input_ids_a, attention_a, input_ids_b, attention_b)
            else:
                input_ids, attention_mask, labels = batch
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

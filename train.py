import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.dataset import TweetDataset
from src.model import MultiTaskClassifier
from src.utils import ensure_dir, load_config, set_seed, write_json


def evaluate(model, loader, device):
    model.eval()
    aggression_true, aggression_pred = [], []
    offense_true, offense_pred = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            aggression_labels = batch["aggression"].to(device)
            offense_labels = batch["offense"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            aggression_logits = outputs["aggression"]
            offense_logits = outputs["offense"]

            aggression_true.extend(aggression_labels.cpu().tolist())
            offense_true.extend(offense_labels.cpu().tolist())
            aggression_pred.extend(aggression_logits.argmax(dim=1).cpu().tolist())
            offense_pred.extend(offense_logits.argmax(dim=1).cpu().tolist())

    return {
        "aggression_accuracy": accuracy_score(aggression_true, aggression_pred),
        "aggression_macro_f1": f1_score(aggression_true, aggression_pred, average="macro"),
        "offense_accuracy": accuracy_score(offense_true, offense_pred),
        "offense_macro_f1": f1_score(offense_true, offense_pred, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the multi-task NLP model.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(42)

    data_path = Path(cfg["data"]["train_path"])
    results_dir = ensure_dir(cfg["outputs"]["results_dir"])
    checkpoint_dir = ensure_dir(cfg["outputs"]["checkpoint_dir"])

    df = pd.read_csv(data_path)
    train_df, val_df = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df["offense"] if df["offense"].nunique() > 1 else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    train_ds = TweetDataset(train_df, tokenizer, max_length=cfg["max_length"])
    val_ds = TweetDataset(val_df, tokenizer, max_length=cfg["max_length"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskClassifier(
        model_name=cfg["model_name"],
        dropout_prob=cfg["dropout_prob"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    aggression_loss = nn.CrossEntropyLoss()
    offense_loss = nn.CrossEntropyLoss()

    logs = []
    best_mean_f1 = -1.0

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            aggression_labels = batch["aggression"].to(device)
            offense_labels = batch["offense"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = aggression_loss(outputs["aggression"], aggression_labels)
            loss = loss + offense_loss(outputs["offense"], offense_labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate(model, val_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = total_loss / max(len(train_loader), 1)
        logs.append(metrics)

        mean_f1 = (metrics["aggression_macro_f1"] + metrics["offense_macro_f1"]) / 2
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            write_json(results_dir / "improved_metrics.json", metrics)

    pd.DataFrame(logs).to_csv(results_dir / "training_log.csv", index=False)


if __name__ == "__main__":
    main()

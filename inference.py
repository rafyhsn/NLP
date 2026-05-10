import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.model import MultiTaskClassifier
from src.utils import load_config


AGGRESSION_LABELS = {
    0: "not_aggressive",
    1: "aggressive",
    2: "highly_aggressive",
}

OFFENSE_LABELS = {
    0: "not_offensive",
    1: "offensive",
}


def predict(text, config_path="config.yaml"):
    cfg = load_config(config_path)
    checkpoint_dir = Path(cfg["outputs"]["checkpoint_dir"])
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Model weights not found at {checkpoint_dir}. Train the model first "
            "or place exported weights in that folder."
        )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = MultiTaskClassifier.from_pretrained(
        checkpoint_dir,
        model_name=cfg["model_name"],
        dropout_prob=cfg["dropout_prob"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    encoded = tokenizer(
        text,
        max_length=cfg["max_length"],
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    aggression_id = int(outputs["aggression"].argmax(dim=1).item())
    offense_id = int(outputs["offense"].argmax(dim=1).item())

    return {
        "text": text,
        "aggression": AGGRESSION_LABELS.get(aggression_id, str(aggression_id)),
        "offense": OFFENSE_LABELS.get(offense_id, str(offense_id)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run inference on one text sample.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    result = predict(args.text, args.config)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

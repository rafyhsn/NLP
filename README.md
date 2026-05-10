# NLP

Multi-task NLP project for aggression and offensive language detection in
Hindi-English code-mixed tweets.

## Project Structure

```text
NLP/
|-- README.md
|-- requirements.txt
|-- train.py
|-- inference.py
|-- config.yaml
|-- data/
|   `-- sample_data.csv
|-- notebooks/
|   `-- 01_inference_demo.ipynb
|-- src/
|   |-- model.py
|   |-- dataset.py
|   `-- utils.py
|-- results/
|   |-- baseline_metrics.json
|   |-- improved_metrics.json
|   `-- training_log.csv
`-- checkpoints/
```

The full dataset and existing tokenizer files are still kept in the project
root. The `checkpoints/` folder is intentionally empty until trained model
weights are exported.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --config config.yaml
```

## Inference

After training or placing model weights in `checkpoints/best_model`, run:

```bash
python inference.py --text "Sample tweet text"
```

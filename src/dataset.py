import torch
from torch.utils.data import Dataset


class TweetDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        encoded = self.tokenizer(
            str(row["tweet"]),
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["aggression"] = torch.tensor(int(row["aggression"]), dtype=torch.long)
        item["offense"] = torch.tensor(int(row["offense"]), dtype=torch.long)
        return item

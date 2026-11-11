import torch
from torch.utils.data import Dataset
from .splits import samples
from ..indexing.index import PAD, EOS


class SequenceDataset(Dataset):
    def __init__(self, split, index, partition, history_length, policy="all_prefixes"):
        if split["item_ids"] != index.item_ids:
            raise ValueError("Split/index catalog row mismatch")
        if split["digest"] != index.payload["provenance"]["split"]:
            raise ValueError("Split/index digest mismatch")
        self.rows = list(samples(split, partition, history_length, policy))
        self.index, self.history_length = index, history_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, row):
        sample = self.rows[row]
        history = torch.full((self.history_length, 4), PAD, dtype=torch.long)
        mask = torch.zeros(self.history_length, dtype=torch.bool)
        length = len(sample["history"])
        history[:length] = torch.from_numpy(self.index.tokens[sample["history"]])
        mask[:length] = True
        label = torch.tensor(self.index.tokens[sample["target"]].tolist() + [EOS])
        return {"history": history, "item_mask": mask, "labels": label,
                "target": sample["target"], "user_id": sample["user_id"],
                "target_position": sample["target_position"]}


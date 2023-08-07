import torch


# Make a simple addition dataset for testing purposes in torch
class AdditionDataset(torch.utils.data.Dataset):

    def __init__(self,
                 length: int,
                 min_val: int = 0,
                 max_val: int = 100,
                 entries: int = 2,
                 seed: int = 42069):
        self.length = length
        self.seed = seed
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

        # pregenerate all floats for the dataset with shape (entries, length) for the range [min_val, max_val)
        self.inputs = torch.randint(min_val,
                                    max_val, (entries, length),
                                    dtype=torch.float32,
                                    generator=self.rng)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "inputs": self.inputs[:, idx],
            "target": torch.sum(self.inputs[:, idx], axis=0, keepdims=True)
        }

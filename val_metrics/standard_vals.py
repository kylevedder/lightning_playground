import torch
from typing import Dict, Any


class L1LossMetric():

    def __init__(self):
        self.reset()

    def update(self, model_forward_result: torch.Tensor,
               input_batch: Dict[str,
                                 Any], batch_idx: int, forward_time: float):

        assert isinstance(
            model_forward_result, torch.Tensor
        ), f"model_forward_result should be a torch.Tensor. Got {type(model_forward_result)}."
        assert isinstance(
            input_batch,
            dict), f"input_batch should be a dict. Got {type(input_batch)}."
        assert isinstance(
            batch_idx,
            int), f"batch_idx should be a int. Got {type(batch_idx)}."
        assert isinstance(
            forward_time, float
        ), f"forward_time should be a float. Got {type(forward_time)}."

        # Get the label and prediction
        label = input_batch["label"]

        # Compute the error
        self.error += torch.sum(torch.abs(label - model_forward_result))
        self.count += label.shape[0]

    def to(self, device):
        self.error = self.error.to(device)
        self.count = self.count.to(device)

    def gather(self, gather_fn):
        error = torch.sum(gather_fn(self.error), axis=0)
        count = torch.sum(gather_fn(self.count), axis=0)
        return {"average_error": error / count, "error": error, "count": count}

    def reset(self):
        self.error = torch.Tensor([0])
        self.count = torch.Tensor([0])

import torch
from typing import Dict, Any


class L1LossMetric():

    def __init__(self):
        self.reset()

    def update(self, input_batch: Dict[str, Any], output_batch: Dict[str, Any],
               batch_idx: int, forward_time: float):

        assert isinstance(
            output_batch,
            dict), f"output_batch should be a dict. Got {type(output_batch)}."
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
        model_forward_result = output_batch["forward"]
        target = input_batch["target"]

        # Compute the error
        self.error += torch.sum(torch.abs(target - model_forward_result))
        self.count += target.shape[0]

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

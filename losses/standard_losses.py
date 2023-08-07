import torch

from typing import Dict, Any


class L1Loss():

    def __init__(self):
        self.loss_fn = torch.nn.L1Loss()

    def __call__(
        self,
        input_batch: Dict[str, Any],
        output_batch: Dict[str, Any],
    ):

        assert isinstance(
            input_batch,
            dict), f"input_batch should be a dict. Got {type(input_batch)}."
        assert isinstance(
            output_batch,
            dict), f"output_batch should be a dict. Got {type(output_batch)}."

        # Get the label and prediction
        target = input_batch["target"]
        model_forward_result = output_batch["forward"]

        assert target.shape == model_forward_result.shape, \
            f"target shape {target.shape} and model_forward_result shape {model_forward_result.shape} should be the same."

        # Compute the loss
        loss = self.loss_fn(model_forward_result, target)

        return {"loss": loss}

import torch


class L1Loss():

    def __init__(self):
        self.loss_fn = torch.nn.L1Loss()

    def __call__(self, input_batch, output_batch):
        assert isinstance(input_batch, dict), "input_batch should be a dict."
        assert isinstance(output_batch, dict), "output_batch should be a dict."

        # Get the label and prediction
        label = input_batch["label"]
        prediction = output_batch["prediction"]

        # Compute the loss
        loss = self.loss_fn(prediction, label)

        return loss
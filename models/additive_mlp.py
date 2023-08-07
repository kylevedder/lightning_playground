import torch


class AdditiveMLP(torch.nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x):
        inputs = x["inputs"]
        outputs = self.sequential(inputs)
        return {"forward": outputs}

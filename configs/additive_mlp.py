dataset = dict(name="AdditionDataset", args=dict(length=1000, seed=42069))
dataloader = dict(
    args=dict(batch_size=16, num_workers=16, shuffle=True, pin_memory=False))

test_dataset = dict(name="AdditionDataset", args=dict(length=1000, seed=1))
test_dataloader = dict(
    args=dict(batch_size=16, num_workers=16, shuffle=True, pin_memory=False))

model = dict(name="AdditiveMLP", args=dict(input_dim=2, output_dim=1))

metric = dict(name="L1LossMetric", args=dict())
learning_rate = 2e-6
save_every = 10
validate_every = 10
epochs = 1000

loss_fn = dict(name="L1Loss", args=dict())
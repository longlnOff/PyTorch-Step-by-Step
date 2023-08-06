
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# learning rate
lr = 1e-3

torch.manual_seed(42)

# model
model = nn.Sequential(nn.Linear(1, 1)).to(device)

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# loss function
loss_fn = nn.MSELoss(reduction='mean')

# train step
train_step = make_train_step(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer
)

# val step
val_step = make_val_step(
    model=model,
    loss_fn=loss_fn
)


# Create SummaryWriter to interface with TensorBoard
writer = SummaryWriter('../runs/simple_linear_regression')

# Fetches a sungle mini-batch of data so we can add graph
x, y = next(iter(train_loader))
writer.add_graph(model, x.to(device))

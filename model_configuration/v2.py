

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set learning rate
lr = 1e-3

torch.manual_seed(42)

# Now we can create a model and send it at once to the device
model = nn.Sequential(nn.Linear(1,1)).to(device=device)

# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Define loss function
loss_fn = nn.MSELoss(reduction='mean')

# Create train step
tran_step = make_train_step(model, loss_fn, optimizer)

# Create validation step
val_step = make_val_step(model, loss_fn)

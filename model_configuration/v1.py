
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set learning parameters
learning_rate = 1e-3

torch.manual_seed(42)

# Create model and send to device
model = nn.Sequential(nn.Linear(1,1)).to(device)

# Defines a SGD optimizer to update the parameters
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Defines a MSE loss function
loss_fn = nn.MSELoss(reduction='mean')

# Creates the training_step function for our model, loss function and optimizer
train_step = make_train_step(model, loss_fn, optimizer)

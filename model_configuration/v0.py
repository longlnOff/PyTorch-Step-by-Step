
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set learning rate
learning_rate = 1e-3

torch.manual_seed(42)

# Create model and send it to device
model = torch.nn.Sequential(torch.nn.Linear(1,1)).to(device)

# Define SGD optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Define loss function
loss_fn = torch.nn.MSELoss(reduction='mean')

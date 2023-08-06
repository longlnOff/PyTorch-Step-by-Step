
torch.manual_seed(0)

# Builds tensors from numpy arrays BEFORE splitting into train and test
x_train_tensor = torch.as_tensor(x_train).float()
y_train_tensor = torch.as_tensor(y_train).float()

# Builds dataset containing ALL data points
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

# Perform split
ratio = 0.8
n_total = len(train_dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train
train_dataset, val_dataset = random_split(train_dataset, [n_train, n_val])

# Builds data loaders
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=64,
    shuffle=False)


true_b = 1
true_w = 2
N = 100

# Data Generation
x = np.random.rand(N, 1)
epsilon = (0.1 * np.random.rand(N, 1))
y = true_b + true_w * x + epsilon

idx = np.arange(N)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:int(N*0.8)]
# Uses remaining indices for validation
val_idx = idx[int(N*0.8):]

# Generate train and validatrion sets
x_train, y_train = x[train_idx], y[train_idx]
x_val = x[val_idx], y[val_idx]

x_train_tensor = torch.as_tensor(x_train).float().to(device=device)
y_train_tensor = torch.as_tensor(y_train).float().to(device=device)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x_train_tensor = torch.as_tensor(x_train).float().to(device=device)
y_train_tensor = torch.as_tensor(y_train).float().to(device=device)

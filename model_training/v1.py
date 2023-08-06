
# Defines number of epochs
n_epochs = 1000

losses = []

# Training loop
for epoch in range(n_epochs):
    loss = train_step(x_train_tensor, y_train_tensor)
    losses.append(loss)

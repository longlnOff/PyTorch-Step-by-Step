
# Defines n_epochs
n_epochs = 1000

for epoch in range(n_epochs):
    # Set model to TRAIN mode
    model.train()

    # Step 1 - Forward pass
    y_pred = model(x_train_tensor)

    # Step 2 - Compute Loss
    loss = loss_fn(y_pred, y_train_tensor)

    # Step 3 - Compute gradients
    loss.backward()

    # Step 4 - Update parameters using gradients and the learning rate
    optimizer.step()
    optimizer.zero_grad()

print(list(model.parameters()))

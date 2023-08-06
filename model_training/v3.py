
# Defines number of epochs
n_epochs = 1000

losses = []

for epoch in range(n_epochs):
    # Training
    loss = mini_batch(device=device,
                      data_loader=train_loader,
                      step=train_step)

    losses.append(loss)

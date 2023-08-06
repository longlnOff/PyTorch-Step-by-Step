
# defines number of epochs
n_epochs = 200

losses = []
val_losses = []

for epoch in range(n_epochs):
    # Training 
    loss = mini_batch(device, train_loader, train_step)
    losses.append(loss)

    # VALIDATION - no gradient tracking needed
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step)
        val_losses.append(val_loss)

    # Records both losses for each epoch under tag 'loss'
    writer.add_scalars(
                        main_tag='loss', 
                        tag_scalar_dict = {'train': loss, 'val': val_loss}, 
                        global_step=epoch)

# close writer
writer.close()

class StepByStep(object):
    def __init__(self, model, loss_fn, optimizer):
        # Here we define the attributes of our class
        # We start by storing the arguments as attributes to use them later
        # arguments
        self.model      = model
        self.loss_fn    = loss_fn
        self.optimizer  = optimizer
        self.device     = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Let's send the model to the specified device right away
        self.model.to(self.device)

        # placeholders for later
        self.train_loader   = None
        self.val_loader     = None
        self.writer         = None

        # variables
        self.train_losses   = []
        self.val_losses     = []
        self.total_epochs   = 0

        # Creates the train_step function for our model, loss function and optimizer
        self.train_step_fn  = self._make_train_step_fn()
        # Creates the val_step function for our model and loss function
        self.val_step_fn    = self._make_val_step_fn()


    def to(self, device):
        # This method allows the user to specify a different device
        # It sets the corresponding attribute (to be used later in
        # the mini-batches) and sends the model to the device
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Couldn't send it to {device}, sending it to {self.device} instead")
            self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        # This method allows the user to define which data loaders to use
        # Both loaders are then assigned to attributes of the class
        # So they can be referred to later
        self.train_loader = train_loader
        self.val_loader   = val_loader

    def set_tensorboard(self, name, folder='runs'):
        # This method allows the user to specify a name for SummaryWriter
        # to interface with TensorBoard
        suffix = datetime.now().strftime("%y%m%d_%H%M%S")
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

    def _make_train_step_fn(self):
        # Builds function that performs a step in the train loop
        def perform_train_step_fn(x, y):
            # Set model to TRAIN mode
            self.model.train()

            # Step 1 - Computes our model's predicted output - forward pass
            yhat = self.model(x)

            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y)

            # Step 3 - Computes gradients for both model's parameters and loss
            loss.backward()

            # Step 4 - Updates parameters using gradients and the chosen optimizer
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Returns the loss
            return loss.item()
        
        # Returns the function that will be called inside the train loop
        return perform_train_step_fn
    
    def _make_val_step_fn(self):
        # Builds function that performs a step in the validation loop
        def perform_val_step_fn(x, y):
            # Set model to EVAL mode
            self.model.eval()

            # Step 1 - Computes our model's predicted output - forward pass
            yhat = self.model(x)

            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y)

            # Returns the loss and the predictions
            return loss.item()
        
        # Returns the function that will be called inside the val loop
        return perform_val_step_fn
    
    def _mini_batch(self, validation=False):
        # The mini-batch method can be used with both loaders
        # The argument 'validation' defines which loader and 
        # corresponding step function will be used
        if validation:
            data_loader = self.val_loader
            step_fn     = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn     = self.train_step_fn

        # Initializes an empty list to accumulate the losses
        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)
        
        loss = np.mean(mini_batch_losses)

        return loss


    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        torch.manual_seed(seed)
        np.random.seed(seed)


    def train(self, n_epochs, seed=42):
        # To ensure reproducibility of the training process
        self.set_seed(seed)

        for epoch in range(n_epochs):
            # Keeps track of the number of epochs
            # by updating the corresponding attribute
            self.total_epochs += 1

            # Training loop
            train_loss = self._mini_batch(validation=False)
            self.train_losses.append(train_loss)

            # Validation loop
            if self.val_loader is not None:
                with torch.no_grad():
                    val_loss = self._mini_batch(validation=True)
                    self.val_losses.append(val_loss)

            if self.writer is not None:
                scalars = {'training': train_loss}
                if val_loss is not None:
                    scalars.update({'validation': val_loss})
                # Records both losses to tensorboard for each epoch under tag "loss"
                self.writer.add_scalars(
                    main_tag='loss',
                    tag_scalar_dict=scalars,
                    global_step=epoch
                )
            
        if self.writer is not None:
            # Fflushes the writer
            self.writer.flush()


    def save_checkpoint(self, filename):
        # Builds dictionary with all elements for resuming training
        checkpoint = {
            'epoch': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        torch.save(checkpoint, filename)


    def load_checkpoint(self, filename):
        # Loads dictionary
        checkpoint = torch.load(filename)

        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_epochs = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses   = checkpoint['val_losses']
        self.model.train()      # Make sure model is in train mode


    def predict(self, x):
        # Set model to EVAL mode
        self.model.eval()

        # Send input to device and computes predictions
        x = x.to(self.device)
        yhat = self.model(x)

        # Set model back to TRAIN mode
        self.model.train()
        
        # Returns predictions
        return yhat.detach().cpu().numpy()


    def plot_losses(self):
        fig = plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training loss', c='b')
        if self.val_loader is not None:
            plt.plot(self.val_losses, label='Validation loss', c='r')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()

        return fig
    

    def add_graph(self):
        if self.train_loader is not None and self.writer is not None:
            # Gets a sample input
            x, _ = next(iter(self.train_loader))
            x = x.to(self.device)

            # Adds model to tensorboard
            self.writer.add_graph(self.model, x)

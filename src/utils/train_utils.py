def train_model(model, loss_fun, optimizer, dataloader, num_epochs):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0

        for batch in dataloader:

            # Forward pass
            outputs = model(batch)
            loss = loss_fun(outputs, [batch.y[:,:6], batch.y[:,6:]])

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    print("Training complete!")

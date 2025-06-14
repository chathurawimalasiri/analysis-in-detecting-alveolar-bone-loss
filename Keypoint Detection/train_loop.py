# ==== Main Training Loop ====

model.train()
n_epochs = 1  # Increase as needed
train_losses, val_losses = [], []
best_val_loss = float('inf')
patience = 10
patience_counter = 0
best_model_state = None
early_stopped = False

for epoch in range(n_epochs):
    # ---- Training ----
    total_train_loss = 0.0
    for i, batch in enumerate(data_loader_train):
        N = len(data_loader_train)
        loss, losses = train_batch(batch, model, optimizer)
        print(f"Epoch {epoch+1}, Batch {i+1}/{N}, Train Loss: {loss.item()}")
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(data_loader_train)
    train_losses.append(avg_train_loss)

    # ---- Validation ----
    total_val_loss = 0.0
    for i, batch in enumerate(data_loader_val):
        N = len(data_loader_val)
        loss, losses = validate_batch(batch, model)
        print(f"Epoch {epoch+1}, Batch {i+1}/{N}, Validation Loss: {loss.item()}")
        total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(data_loader_val)
    val_losses.append(avg_val_loss)

    # Learning rate scheduler
    lr_scheduler.step()

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
        print(f"Best model at epoch {epoch+1}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}.")
            early_stopped = True
            break

# Restore best model if early stopped
if early_stopped and best_model_state is not None:
    model.load_state_dict(best_model_state)

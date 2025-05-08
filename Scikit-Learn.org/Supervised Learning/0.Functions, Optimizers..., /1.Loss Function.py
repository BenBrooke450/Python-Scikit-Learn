import numpy as np
# Data
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])
# Initialize parameters
w = 0.0
b = 0.0
# Hyperparameters
learning_rate = 0.01
epochs = 150
# SGD Training
for epoch in range(epochs):
    total_loss = 0
    
    for i in range(len(x)):
        # Step 1: Predict
        y_pred = w * x[i] + b
        # Step 2: Compute error and loss
        error = y_pred - y[i]
        loss = error ** 2
        total_loss += loss
        # Step 3: Compute gradients
        grad_w = 2 * error * x[i]
        grad_b = 2 * error
        # Step 4: Update parameters
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
    # Print info every 5 epochs
    if epoch % 5 == 0:
        print(f"Epoch {epoch:2d}: Loss = {total_loss:.4f}, w = {w:.4f}, b = {b:.4f}")
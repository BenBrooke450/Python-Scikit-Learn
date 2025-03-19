

import numpy as np
import matplotlib.pyplot as plt


# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

print(f"x_train = {x_train}")
#x_train = [1. 2.]


print(f"y_train = {y_train}")
#y_train = [300. 500.]


# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")



m = x_train.shape[0]
print(f"Number of training examples is: {m}")
#Number of training examples is: 2



# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')

# Set the title
plt.title("Housing Prices")

# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')


# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()








import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # USED IN METHOD 3 ONLY
from sklearn.model_selection import train_test_split # USED IN PLOT


# Read data into dataframe
df = pd.read_csv('./Dataset/housing.csv')
df = df.dropna() # Remove missing data
print(df.head())

# Preprocess (Remove Col 10)
X = df.drop(['ocean_proximity', 'median_house_value'], axis=1)
Y = df[['median_house_value']]
print(X.head())
print(Y.head())
print()

X = X.to_numpy(dtype=float)
Y = Y.to_numpy(dtype=float).flatten()
print(X, Y)
print(X.shape, Y.shape)
print()

# Standardize
# z = x - mu / std
X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
print(f'X_Scaled: \n{X}')
Y = (Y - np.mean(Y)) / np.std(Y)
print(f'Y_Scaled: \n{Y}')

# Methods

# Method OLS
print(' METHOD 1 : 0LS\n')
# Add baseline [1, X1, X2..., Xd]
N = X.shape[0] # Grabs number of rows
print(f'Shape before baseline: {X.shape}')
X_aug = np.hstack([np.ones((N, 1)), X]) # d + 1
print(X_aug)
print(f' Shape after: {X_aug.shape}')

# Add theta/B^ (parameters)
total_weights = X_aug.shape[1]  # Grabs number of columns
theta = np.zeros(total_weights) # initialize parameters to zero
print(theta, theta.shape)

# theta = (X^T X)^(-1) X^T y
theta = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ Y
print(theta.shape)
print(theta) # All of our weights (theta) for each feature

# Generate Predictions for mean_house_value: y^ = XB^
Y_pred = X_aug @ theta
print(f'Predictions: \n{Y_pred}')
print(Y_pred.shape)

# Compute MSE: (sum(y - y^)^2)/n)
MSE = np.mean((Y_pred - Y) ** 2)
print(f"\nTraining MSE: How far model is off on avg: {np.sqrt(MSE):.4f} (Without SQRT: {MSE})\n")

print('First 10 Predictions vs True:')
for i in range(10):
    print(f'Prediction = {Y_pred[i].item():.2f} Ground_Truth = {Y[i].item():.2f}')

print()

# Method 2
print(' METHOD 2 : GD\n')

# Weights initialization to zero
W = np.zeros(total_weights)
print(W, W.shape)

# Set hyperparameters
learning_rate = 0.01  # Steps to take during weight update
epochs = 2800         # One full pass through entire training dataset

# Train
loss_list = []
for epoch in range(epochs):
    # Generate Predictions for mean_house_value: y^ = XW
    Y_pred = X_aug @ W
    # Difference between Predictions & Ground_truth
    E = Y_pred - Y
    # Calculate loss after each epoch
    MSE = np.mean(E ** 2)
    loss_list.append(MSE)  # Place every MSE loss in the list
    # Calculate Gradients (vector): partial derivatives = 2/N X^T(y^ - y)
    gradients = (2 / N) * (X_aug.T @ E)  # Finds which feature's weight contributed to the error
    # Update weights/params: move in the opposite direction of gradient to decrease error
    W = W - learning_rate * gradients

    # Print loss after every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, MSE = {MSE:.4f}')
print(f'\nFinal: {MSE} (Similar to OLS)')
print()

# Method 3
print(' METHOD 3 : LIBRARY\n')
model = LinearRegression().fit(X, Y)
lib_w = model.coef_
lib_b = model.intercept_
print(f'Models weights: {lib_w}\n')
print(f'Models intercept: {lib_b}\n')
lib_theta = np.append(lib_b, lib_w)
print(lib_theta)

Y_pred_LR = model.predict(X)
print()

print('First 10 Predictions vs True:')
for i in range(10):
    print(f'Prediction = {Y_pred_LR[i].item():.2f} Ground_Truth = {Y[i].item():.2f}')

# Comparisons

# Comparison of Theta Parameters Across Models:
# Grouped bar chart (documentation from matplotlib)
T = ('Theta_0', 'Theta_1', 'Theta_2', 'Theta_3', 'Theta_4',
    'Theta_5', 'Theta_6', 'Theta 7', 'Theta_8')

weights = {'Theta_OLS' : theta,
           'Theta_GD' : W,
           'Theta_Lib': lib_theta}

x = np.arange(len(T))
width = 0.2
multiplier = 0

fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')

for a, w_nums in weights.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, w_nums, width, label=a)
    multiplier += 1

ax.axhline(0, color='black', linewidth=2)

ax.set_ylim(-1.0, 1.0)

ax.set_xticks(x + width)
ax.set_xticklabels(T, rotation=45, ha='right')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(axis='x', which='major', pad=5)

ax.set_ylabel('Value (standardized)')
ax.set_title('Comparison of Theta Parameters Across Models')
ax.legend(loc='upper left', ncols=3, fontsize='small')
plt.show()

# Parameter Difference from best OLS model param to other 2 (bar plot)
OLS_GD = np.abs(theta - W)
OLS_Lib = np.abs(theta - lib_theta)

differences ={
    '|Theta_OLS - Theta_GD|': OLS_GD,
    '|Theta_OLS - Theta_Lib|': OLS_Lib
}

x = np.arange(len(T))
width = 0.2
multiplier = 0

fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')

for a, diffs in differences.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, diffs, width, label=a)
    multiplier += 1

ax.set_xticks(x + width)
ax.set_xticklabels(T, rotation=45, ha='right')

ax.set_ylabel('Absolute Differences (standardized)')
ax.set_title('Parameter Differences')
ax.legend()
plt.show()

# Compare loss value (L2) of three different models for diff training data size

# Lists for all losses
OLS_loss = []
GD_loss = []
Lib_loss = []

# Different training data size {10%, 50%, 100%}
sizes = [0.1, 0.5, 1.0]

for size in sizes:
    # Removed params test
    X_train, _, Y_train, _ = train_test_split(X_aug, Y, test_size=size, random_state=42)

    # L2 = sum(y - y^)^2 (y^ being XB (theta))
    OLS_loss.append(np.mean((X_train @ theta - Y_train) **2))
    GD_loss.append(np.mean((X_train @ W - Y_train) **2))
    Lib_loss.append(np.mean((X_train @ lib_theta - Y_train) **2))

# Dot plot
plt.figure(figsize=(8, 6))
xlabel = ['10%', '50%', '100%']
x = np.arange(len(xlabel))

plt.scatter(x, OLS_loss, color='blue', label='OLS Loss')
plt.scatter(x, GD_loss, color='green', label='GD Loss')
plt.scatter(x, Lib_loss, color='red', label='Lib Loss')

plt.xticks(x, xlabel)
plt.ylabel('Loss')
plt.title('Loss Comparison Across Data Sizes')
plt.legend()
plt.show()
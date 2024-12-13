# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 01:37:27 2024

@author: Tate
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Read the Excel file into a pandas DataFrame
df = pd.read_excel('Noncorrupted Dataset.xlsx', header=None)  # Replace 'your_file.xlsx' with the path to your file

# Convert the DataFrame to a NumPy array
data = df.to_numpy()  # or df.values
np.set_printoptions(precision=10, suppress=True)

print(data)
x = data[:, 0]
y_clean= pd.to_numeric(data[:, 1])

# Add random Gaussian noise to create noisy data
noise = np.random.normal(0, .0000001, y_clean.shape)
y_noisy = y_clean + noise

# # Normalize x and y
# x_normalized = (x - x.min()) / (x.max() - x.min())
# y_normalized = (y_noisy - y_noisy.min()) / (y_noisy.max() - y_noisy.min())
# y_normalized_clean = (y_clean - y_clean.min()) / (y_clean.max() - y_clean.min())

# Combine x and y for input to the autoencoder
data_noisy = np.vstack((x, y_noisy)).T
data_clean = np.vstack((x, y_clean)).T

scaler = MinMaxScaler()
data_noisy_scaled = scaler.fit_transform(data_noisy)
data_clean_scaled = scaler.transform(data_clean)

# Input layer
input_layer = Input(shape=(2,))  # Two features: x and y

# Encoder: Reduce dimensionality
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)

# Latent space
latent = Dense(8, activation='relu')(encoded)

# Decoder: Reconstruct data
decoded = Dense(16, activation='relu')(latent)
decoded = Dense(32, activation='relu')(decoded)
output_layer = Dense(2, activation='linear')(decoded)

# Build and compile the model
autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer=Adam(learning_rate=0.02), loss='mse')

print("x values:", x[:10])
print("y_clean values:", y_clean[:10])
print("y_noisy values:", y_noisy[:10])


# Train the model
history = autoencoder.fit(
    data_noisy_scaled,  # Input (noisy data)
    data_clean_scaled,  # Target (clean data)
    epochs=5000,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Use the model to predict denoised data
data_denoised_scaled = autoencoder.predict(data_noisy_scaled)

# Inverse transform the data to get back original scale
data_denoised = scaler.inverse_transform(data_denoised_scaled)

# Plot the clean, noisy, and autoencoder noisy output data
plt.figure(figsize=(12, 6))
plt.plot(x, y_clean, label='Clean Data', color='green', linewidth=2)  # Clean data
plt.scatter(x, y_noisy, label='Noisy Data', color='red', s=5, alpha=0.5)  # Noisy data
plt.plot(data_noisy[:, 0], data_noisy[:, 1], label='Autoencoder Output Before Denoising', color='orange', linewidth=2)  # Autoencoder output before denoising
plt.legend()
plt.xlim(min(x), max(x))  # Use the min and max values of x for x-axis limits
plt.ylim(min(y_clean), max(y_clean))  # Use the min and max values of y for y-axis limits
plt.title('Autoencoder Output Before Denoising Check')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(x, y_clean, label='Clean Data', color='green', linewidth=2)
plt.scatter(x, y_noisy, label='Noisy Data', color='red', s=5, alpha=0.5)
plt.plot(data_denoised[:, 0], data_denoised[:, 1], label='Denoised Data', color='blue', linewidth=2)
plt.legend()
plt.xlim(min(x), max(x))  # Use the min and max values of x for x-axis limits
plt.ylim(min(y_clean), max(y_clean))  # Use the min and max values of y for y-axis limits
plt.title('Denoising Autoencoder Results')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Plot the clean data and noisy data
plt.figure(figsize=(12, 6))
plt.plot(x, y_clean, label='Clean Data', color='green', linewidth=2)  # Clean data
plt.scatter(x, y_noisy, label='Noisy Data', color='red', s=5, alpha=0.5)  # Noisy data
plt.legend()
plt.title('Clean Data vs. Noisy Data')
plt.xlim(min(x), max(x))  # Use the min and max values of x for x-axis limits
plt.ylim(min(y_clean), max(y_clean))  # Use the min and max values of y for y-axis limits
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print("Min and Max of x:", x.min(), x.max())
print("Min and Max of y_clean:", y_clean.min(), y_clean.max())


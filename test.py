import matplotlib.pyplot as plt
import numpy as np

# Constants
C = 100  # Total capacity in bps
alpha = 1  # Additive increase (1 MSS per RTT)
beta1 = 3/4  # Multiplicative decrease for R1
beta2 = 1/4  # Multiplicative decrease for R2
tolerance = 1  # Tolerance for convergence

# Initial flow rates
R1, R2 = 0, 2

# Lists to store flow rates over time
R1_values, R2_values = [], []

while not (abs(R1 - 3/4 * C) < tolerance and abs(R2 - 1/4 * C) < tolerance):
    # Additive Increase
    R1 += alpha
    R2 += alpha

    # Check if combined rate exceeds capacity
    if R1 + R2 > C:
        # Multiplicative Decrease
        R1 *= beta1
        R2 *= beta2

    # Store values
    R1_values.append(R1)
    R2_values.append(R2)

# Plotting the results with R2 as the X-axis and R1 as the Y-axis
plt.figure(figsize=(10, 10))
plt.scatter(R2_values, R1_values, c=np.arange(len(R1_values)), cmap='viridis')
plt.colorbar(label='Iterations')

# Capacity line (C) from (0,C) to (C,0)
plt.plot([0, C], [C, 0], 'r--', label='Capacity Line')

# Highlighting the last point
last_R1, last_R2 = R1_values[-1], R2_values[-1]
plt.scatter(last_R2, last_R1, color='red', s=100)  # Larger and different color for the last point
plt.annotate(f'({last_R2:.2f}, {last_R1:.2f})', (last_R2, last_R1), textcoords="offset points", xytext=(10,-10), ha='center')

plt.xlabel('Flow Rate R2')
plt.ylabel('Flow Rate R1')
plt.title('AIMD Flow Rates R1 vs. R2')
plt.xlim(0, C)
plt.ylim(0, C)
plt.grid(True)
plt.legend()
plt.show()

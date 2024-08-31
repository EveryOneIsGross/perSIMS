import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Read the CSV file
df = pd.read_csv('simulation_metrics.csv')

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.ndimage import gaussian_filter

# Read the CSV file
df = pd.read_csv('simulation_metrics.csv')

# 1. Sim's 3D Location Density Plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Create a 2D histogram
hist, x_edges, y_edges = np.histogram2d(df['x'], df['y'], bins=(50, 50))

# Apply Gaussian smoothing to the histogram
smoothed_hist = gaussian_filter(hist, sigma=1)

# Get the centers of the bins
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2

# Create a meshgrid
X, Y = np.meshgrid(x_centers, y_centers)

# Flatten the arrays
x_flat = X.flatten()
y_flat = Y.flatten()
density_flat = smoothed_hist.T.flatten()  # Transpose the histogram

# Remove zero density points
mask = density_flat > 0
x_plot = x_flat[mask]
y_plot = y_flat[mask]
density_plot = density_flat[mask]

# Normalize density for color mapping
norm = plt.Normalize(density_plot.min(), density_plot.max())

# Create the 3D scatter plot
scatter = ax.scatter(x_plot, y_plot, density_plot, 
                     c=density_plot, cmap='viridis', 
                     s=density_plot/density_plot.max()*100,  # Adjust size scaling as needed
                     alpha=0.7, norm=norm)

# Add a color bar
cbar = fig.colorbar(scatter, shrink=0.5, aspect=5)
cbar.set_label('Activity Density', rotation=270, labelpad=15)

ax.set_title("Sim's 3D Location Density Plot")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Density')

# Adjust the view angle for better visibility
ax.view_init(elev=30, azim=45)

plt.savefig('sim_3d_location_density_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Financial Trends
plt.figure(figsize=(12, 6))
plt.plot(df['step'], df['money'])
plt.title("Sim's Financial Trend")
plt.xlabel('Steps')
plt.ylabel('Money')
plt.tight_layout()
plt.savefig('financial_trend.png')
plt.close()

# 3. Needs Over Time
needs = [col for col in df.columns if col.startswith('need_')]
plt.figure(figsize=(12, 6))
for need in needs:
    plt.plot(df['step'], df[need], label=need.replace('need_', ''))
plt.title("Sim's Needs Over Time")
plt.xlabel('Steps')
plt.ylabel('Need Level')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('needs_over_time.png')
plt.close()

# 4. Mood Distribution
plt.figure(figsize=(8, 6))
df['mood'].value_counts().plot(kind='bar')
plt.title("Distribution of Sim's Moods")
plt.xlabel('Mood')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('mood_distribution.png')
plt.close()

# 5. Market Trends
market_items = set([col.split('_')[1] for col in df.columns if col.startswith('market_') and col.endswith('_price')])
for item in market_items:
    plt.figure(figsize=(12, 6))
    plt.plot(df['step'], df[f'market_{item}_price'], label='Price')
    plt.plot(df['step'], df[f'market_{item}_quantity'], label='Quantity')
    plt.title(f"Market Trends for {item}")
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'market_trends.png')
    plt.close()

# 6. Activity Duration Distribution
plt.figure(figsize=(10, 6))
df['activity_duration'].hist(bins=20)
plt.title("Distribution of Activity Durations")
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('activity_duration_distribution.png')
plt.close()

# 7. 3D Trajectory Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(df['x'], df['y'], df['step'])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Steps')
ax.set_title("Sim's 3D Trajectory Over Time")
plt.tight_layout()
plt.savefig('sim_3d_trajectory.png')
plt.close()

print("Visualization complete. Check the following files in the current directory:")
print("needs_over_time.png")
print("mood_distribution.png")
print("market_trends.png")
print("activity_duration_distribution.png")
print("sim_3d_trajectory.png")
print("sim_3d_location_density_plot.png")

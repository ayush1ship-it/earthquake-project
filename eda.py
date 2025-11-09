import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
#df = pd.read_csv('Japan_processed.csv')
df = pd.read_csv('USGS_processed_2.csv')
print(df.head())

# Checking for missing values
missing_per_column = df.isnull().sum()
print(missing_per_column)

# Histogram for Distribution of Earthquake Magnitudes (ALL)
#----------------------------------------------------------
plt.hist(df['Magnitude'], bins=np.arange(2.5, 9.0, 0.25), edgecolor='black')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.title('Distribution of Earthquake Magnitudes')
plt.show()

# Histogram for Distribution of Earthquake Magnitudes (ABOVE 6.0)
#----------------------------------------------------------------
plt.hist(df['Magnitude'], bins=np.arange(6.0, 9.0, 0.25), edgecolor='black')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.title('Distribution of Earthquake Magnitudes [Magnitude 6.0 and above]')
plt.show()

# Plotting average mag per month and year
#-----------------------------------------
# Group the data by month and year and calculate the mean and median for mag and magError
monthly_stats = df.groupby('Month')[['Magnitude']].agg(['mean'])
yearly_stats = df.groupby('Year')[['Magnitude']].agg(['mean'])
plt.figure(figsize=(8, 5))

# First plot: Average mag per month
plt.subplot(1, 2, 1)
plt.plot(monthly_stats.index, monthly_stats['Magnitude']['mean'], label='Avg Magnitude (Mean)', marker='o')
x_ticks = np.arange(1, 13, 1)
plt.xticks(ticks = x_ticks)
plt.title('Average Magnitude per Month')
plt.xlabel('Month')
plt.ylabel('Avg Magnitude')
plt.legend()

# Second plot: Average mag per year with mean and median
plt.subplot(1, 2, 2)
plt.plot(yearly_stats.index, yearly_stats['Magnitude']['mean'], label='Avg Magnitude (Mean)', marker='o')
x_ticks = np.arange(2011, 2026, 2)
plt.xticks(ticks = x_ticks)
plt.title('Average Magnitude per Year')
plt.xlabel('Year')
plt.ylabel('Avg Magnitude')
plt.legend()

plt.tight_layout()
plt.show()

# Plotting earthquake locations
#------------------------------
plt.figure(figsize=(8, 6))

# Set base size, and increase only if magnitude > 6.0
sizes = df['Magnitude'].apply(lambda m: 20 if m > 6.0 else 5)

#plt.scatter(df['Longitude'], df['Latitude'], c=df['Magnitude'], cmap='coolwarm', s=20)
plt.scatter(df['Longitude'], df['Latitude'], c=df['Magnitude'], cmap='coolwarm', s=sizes, alpha=0.7)

plt.colorbar(label='Magnitude')
plt.title('Earthquake Locations (Jan 2011 to Jun 2025) - Color-coded by Magnitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

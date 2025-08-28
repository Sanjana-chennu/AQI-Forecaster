import pandas as pd
import matplotlib.pyplot as plt
import os

print("--- Analyzing Historical Pollution Patterns ---")

# --- 1. Load the Data ---
# Use an absolute path to be robust
file_path = os.path.join(os.path.dirname(__file__), 'data', 'LSTM-Multivariate_pollution.csv')
df = pd.read_csv(file_path, parse_dates=['date'])
df.set_index('date', inplace=True)

print(f"Data loaded successfully. Date range: {df.index.min()} to {df.index.max()}")

# --- 2. Select a Time Window to Analyze ---
# Let's look at the first week of data as an example.
start_date = '2010-01-02'
end_date = '2010-01-09'
df_subset = df.loc[start_date:end_date]

print(f"\nPlotting data for the period: {start_date} to {end_date}")

# --- 3. Create the Plot ---
plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style for the plot
plt.figure(figsize=(16, 8))

# Plot the 'pollution' column against the datetime index
plt.plot(df_subset.index, df_subset['pollution'], marker='.', linestyle='-', markersize=4)

# --- 4. Format the Plot for Clarity ---
plt.title(f'Hourly PM2.5 Pollution from {start_date} to {end_date}', fontsize=16)
plt.xlabel('Date and Time', fontsize=12)
plt.ylabel('PM2.5 Concentration', fontsize=12)
plt.xticks(rotation=45) # Rotate date labels for better readability
plt.tight_layout() # Adjust layout to make room for labels

# Save the plot to a file
plot_path = 'historical_pattern_analysis.png'
plt.savefig(plot_path)
print(f"Analysis plot saved to: {plot_path}")

# Display the plot
plt.show()
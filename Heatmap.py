import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

index = ['A1 Expressiveness', 'A2 Statistical Learning', 'A3 Optimization Dynamics',
         'A4 Implicit Programs', 'A5 Cognitive Inspiration', 'A6 Information Theory']
columns = ['S5', 'Circuit', 'GSM8K', 'StrategyQA', 'CoinFlip', 'MultiArith',
           'AQuA', 'CSQA', 'LastLetter', 'Parity', 'ModularAdd']
data = [
    [3.9, 5.3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.90, 0.04],
    [np.nan, np.nan, 0.86, np.nan, np.nan, np.nan, 0.45, np.nan, np.nan, np.nan, np.nan],
    [np.nan, np.nan, np.nan, np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, 0.04],
    [3.9, 5.3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [np.nan, np.nan, np.nan, 0.30, 0.90, np.nan, np.nan, np.nan, 7.5, np.nan, np.nan],
    [np.nan, np.nan, 0.86, np.nan, np.nan, np.nan, np.nan, 0.17, np.nan, np.nan, np.nan]
]

df = pd.DataFrame(data, index=index, columns=columns)

# Handling Missing Values: Replace NaN with a Special Value
special_value = -999
df_filled = df.fillna(special_value)

# Create a mask (for hiding annotations of special values)
mask = df.isna()

# Obtain the original color mapping
original_cmap = sns.color_palette("YlGnBu", as_cmap=True)

# Create a new color map that maps special values to light gray
colors = list(original_cmap(np.linspace(0, 1, 256)))
colors.insert(0, (0.94, 0.94, 0.94))
new_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# Define the color mapping range
vmin = df.min().min()
vmax = df.max().max()

plt.figure(figsize=(12, 6))

sns.heatmap(df_filled,
            annot=True,
            fmt=".2f",
            cmap=new_cmap,
            cbar_kws={'label': 'Relative Improvement (w)'},
            linewidths=1.5,
            linecolor='gray',
            mask=mask,
            vmin=vmin,
            vmax=vmax
            )

plt.title("Heatmap of CoT Explanation Angles × Tasks\n(Weight = (Acc_CoT - Acc_no) / Acc_no)")
plt.xlabel("Tasks")
plt.ylabel("Explanation Angles")
plt.tight_layout()
plt.show()

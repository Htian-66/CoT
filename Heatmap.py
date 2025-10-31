import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

index = ['A1 Expressiveness', 'A2 Statistical Learning', 'A3 Optimization Dynamics',
         'A4 implicit vertical reasoning', 'A5 Cognitive Inspiration', 'A6 Information Theory']
columns = ['Circuit', 'GSM8K', '5×5 Multiplication', 'CoinFlip', 'MultiArith',
           'AQuA', 'GSM8K-Aug', 'LastLetter', 'Parity', 'Llama-3-8B Arithmetic']
data = [
    [ 5.3, np.nan, np.nan, np.nan, 3.0, np.nan, np.nan, 7.5, 0.90, np.nan],
    [ np.nan, 0.86, np.nan, np.nan, np.nan, 0.45, np.nan, np.nan, np.nan, np.nan],
    [ np.nan, np.nan, np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, np.nan],
    [np.nan, np.nan, 49.7, np.nan, np.nan, np.nan, 0.29, np.nan, np.nan, np.nan],
    [np.nan, np.nan, np.nan, 0.90, np.nan, np.nan, np.nan, 7.5, np.nan, np.nan],
    [np.nan, 0.86, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.90]
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

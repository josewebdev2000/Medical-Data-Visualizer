import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' column
df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)
df["overweight"] = (df["BMI"] > 25).astype(int)
df.drop(["BMI"], axis = 1, inplace = True)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
def normalize(value):
    if value == 1:
        return 0
    elif value > 1:
        return 1
    else:
        return value

df["cholesterol"] = df["cholesterol"].apply(normalize)
df["gluc"] = df["gluc"].apply(normalize)

# Draw Categorical Plot
def draw_cat_plot():
    # Convert data to long format
    df_long = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # Create a catplot
    sns.set(style="whitegrid")
    fig = sns.catplot(x="variable", hue="value", col="cardio",
                    data=df_long, kind="count")

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Filter out incorrect data
    df_cleaned = df[
        (df['ap_lo'] <= df['ap_hi']) &  # diastolic pressure is not higher than systolic
        (df['height'] >= df['height'].quantile(0.025)) &  # height >= 2.5th percentile
        (df['height'] <= df['height'].quantile(0.975)) &  # height <= 97.5th percentile
        (df['weight'] >= df['weight'].quantile(0.025)) &  # weight >= 2.5th percentile
        (df['weight'] <= df['weight'].quantile(0.975))    # weight <= 97.5th percentile
    ]

    # Create a correlation matrix
    corr_matrix = df_cleaned.corr()

    # Mask the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, mask=mask, cmap='RdBu', vmax=0.3, vmin=-0.1,
                square=True, annot=True, fmt='.1f', linewidths=.5, ax=ax)

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig

# Create and save the graphics
draw_cat_plot()
draw_heat_map()
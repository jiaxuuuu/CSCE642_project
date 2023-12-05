import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm

def plot_normalized_distribution(df, columns, normalize):
    """
    Plots the normalized (standardized) normal distribution for specified columns in a DataFrame.
    
    :param df: DataFrame containing the data
    :param columns: List of column names to plot the distribution for
    :param normalize: Boolean to indicate whether to normalize the data
    """
    num_columns = len(columns)
    plt.figure(figsize=(num_columns * 6, 6))

    for i, column in enumerate(columns, start=1):
        plt.subplot(1, num_columns, i)
        
        # data normalization
        if normalize:
            data = (df[column] - df[column].mean()) / df[column].std()
            mean, std = 0, 1
        else:
            data = df[column]
            mean, std = df[column].mean(), df[column].std()

        # histogram and KDE
        sns.histplot(data, kde=True, stat="density", linewidth=0, label=f"KDE of {column}")

        # normalized dist
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean, std)
        plt.plot(x, p, 'k', linewidth=2, label="Normal Dist")

        plt.title(f"{'Normalized' if normalize else ''} Distribution for {column}")
        plt.ylabel("Density")
        plt.legend()

    plt.tight_layout()
    plt.show()
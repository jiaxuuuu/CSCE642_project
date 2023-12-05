import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.stats import gaussian_kde

def calculate_Q_values(df, unit_rwd, rwd, gamma, module_column_list):
    """
    Calculate Q values for a given module.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    unit_rwd (float): Unit reward for the module.
    rwd (float): Weight of the reward.
    gamma (float): Discount factor.
    module_column_list (list): List of columns to use for Q value calculation.

    Returns:
    pd.DataFrame: DataFrame with calculated Q values.
    """
    df_selected = df.filter(module_column_list)
    
    if len(module_column_list) != 0:
        df_selected['Q'] = sum(unit_rwd * rwd * np.power(gamma, df_selected[col]) for col in module_column_list)
    else:
        df_selected['Q'] = np.zeros(len(df_selected))
        print("Module column list is empty. df['Q'] has been set to 0")

    return df_selected

def create_rbf_heatmap(df, ax, rwd, gamma, smooth):
    """
    Create a heatmap using Radial Basis Function (RBF) interpolation.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    ax (matplotlib.axes.Axes): Axes object to draw the heatmap.
    rwd (float): Reward value.
    gamma (float): Gamma value.
    smooth (float): Smoothing parameter for RBF.
    """
    # Extracting x, y, z coordinates from the 'Position' column.
    position_components = df['Position'].str.split(expand=True)
    x, y, z = position_components.iloc[:, 2].astype(float), position_components.iloc[:, 0].astype(float), df['Q']

    rbf = Rbf(x, y, z, function='linear', smooth=smooth)

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = rbf(xi, yi)

    aspect_ratio = (x.max() - x.min()) / (y.max() - y.min())

    im = ax.imshow(zi, aspect=aspect_ratio, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
    ax.scatter(x, y, c=z)
    plt.colorbar(im, ax=ax, label='Q Value')
    ax.set_xlabel('Agent X Position')
    ax.set_ylabel('Agent Z Position')
    ax.set_title(f'RBF Interpolated Q Values (Rwd: {rwd}, Gamma: {gamma})')

def create_kde_heatmap(df, ax, rwd, gamma):
    """
    Create a heatmap using Kernel Density Estimation (KDE).

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    ax (matplotlib.axes.Axes): Axes object to draw the heatmap.
    rwd (float): Reward value.
    gamma (float): Gamma value.
    """
    position_components = df['Position'].str.split(expand=True)
    x, y, z = position_components.iloc[:, 2].astype(float), position_components.iloc[:, 0].astype(float), df['Q']

    kde = gaussian_kde([x, y], weights=z)

    xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

    aspect_ratio = (x.max() - x.min()) / (y.max() - y.min())

    im = ax.imshow(zi, aspect=aspect_ratio, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
    plt.colorbar(im, ax=ax, label='KDE Density')
    ax.set_xlabel('Agent X Position')
    ax.set_ylabel('Agent Z Position')
    ax.set_title(f'KDE of Q Values (Rwd: {rwd}, Gamma: {gamma})')

def create_all_heatmaps(df_selected_3, nrows, ncols, rwd_list, gamma_list, smooth):
    """
    Create and display all heatmaps.

    Args:
    df_selected_3 (pd.DataFrame): DataFrame containing the data.
    nrows (int): Number of rows in the subplot.
    ncols (int): Number of columns in the subplot.
    rwd_list (list): List of reward values.
    gamma_list (list): List of gamma values.
    smooth (float): Smoothing parameter for RBF.
    """
    _, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5))

    create_rbf_heatmap(df_selected_3, axs[0], rwd_list[0], gamma_list[0], smooth)
    create_kde_heatmap(df_selected_3, axs[1], rwd_list[0], gamma_list[0])

    plt.tight_layout()
    plt.show()

def visualize_heatmaps(df_path, unit_rwd_list, rwd_list, gamma_list, m1_col_list, m2_col_list, smooth):
    """
    Visualize heatmaps for Q values.

    Args:
    df_path (str): Path to the data file.
    unit_rwd_list (list): List of unit rewards.
    rwd_list (list): List of rewards.
    gamma_list (list): List of gamma values.
    m1_col_list (list): List of columns for module 1.
    m2_col_list (list): List of columns for module 2.
    smooth (float): Smoothing parameter for RBF.
    """
    df = pd.read_csv(df_path)
    df_selected_1 = calculate_Q_values(df, unit_rwd_list[0], rwd_list[0], gamma_list[0], m1_col_list)
    df_selected_2 = calculate_Q_values(df, unit_rwd_list[1], rwd_list[1], gamma_list[1], m2_col_list)

    df_selected_3 = df.copy()
    df_selected_3['Q'] = df_selected_1['Q'] + df_selected_2['Q']

    create_all_heatmaps(df_selected_3, 1, 2, rwd_list, gamma_list, smooth)

# Example usage (make sure to replace 'agg_results' with your actual results object)
def heatmap_vis_integrated(results, index, mode):
    """
    Integrate and visualize heatmaps.

    Args:
    results (list): List of optimization results.
    index (int): Index of the result to visualize.
    mode (str): Mode ('agg' or 'con') indicating the type of data.
    """
    rwd1, gamma1, rwd2, gamma2 = results[index].x
    rwd_list = [round(rwd1, 2), round(rwd2, 2)]
    gamma_list = [round(gamma1, 2), round(gamma2, 2)]

    m1_col_list = ['d2target']
    m2_col_list = [' d2building', ' d2obstacle1', ' d2obstacle2', ' d2obstacle3', ' d2obstacle4', ' d2obstacle5']

    filepath = f'Record/Aggressive/Data/heatmapcsv/{index}heatmap.csv' if mode == 'agg' else f'Record/Conservative/Data/heatmapcsv/{index}heatmap.csv'

    visualize_heatmaps(filepath, [-1, -1], rwd_list, gamma_list, m1_col_list, m2_col_list, 0.1)

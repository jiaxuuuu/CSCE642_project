import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf # griddata
from scipy.stats import gaussian_kde

"""
Description:
특정 모듈의 인스턴스들로 이루어진 데이터프레임을 받아서 해당 모듈의 Q(s,a)를 계산 후 시각화합니다.
"""

def calculate_Q_values(df, unit_rwd, rwd, gamma, module_column_list):
    df_selected = df.filter(module_column_list)
    
    # 모든 컬럼에 대해 Q 값 계산
    df_selected['Q'] = sum(unit_rwd * rwd * np.power(gamma, df_selected[col]) for col in module_column_list)

    return df_selected

def create_rbf_heatmap(df, ax, rwd, gamma, smooth):
    x, y, z = df.iloc[:, 0], df.iloc[:, 1], df['Q']
    rbf = Rbf(x, y, z, function='multiquadric', smooth=smooth)

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = rbf(xi, yi)

    aspect_ratio = (x.max() - x.min()) / (y.max() - y.min())

    im = ax.imshow(zi, aspect=aspect_ratio, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
    ax.scatter(x, y, c=z)
    plt.colorbar(im, ax=ax, label='Q Value')
    ax.set_xlabel('Physical Distance')
    ax.set_ylabel('Temporal Distance')
    ax.set_title(f'RBF. Q Values. R:{rwd}, Gamma:{gamma}') # RBF Interpolated Heatmap of Q Values

def create_kde_heatmap(df, ax, rwd, gamma):
    x, y, z = df.iloc[:, 0], df.iloc[:, 1], df['Q']
    kde = gaussian_kde([x, y], weights=z)

    xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

    aspect_ratio = (x.max() - x.min()) / (y.max() - y.min())

    im = ax.imshow(zi, aspect=aspect_ratio, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
    plt.colorbar(im, ax=ax, label='KDE Density')
    ax.set_xlabel('Physical Distance')
    ax.set_ylabel('Temporal Distance')
    ax.set_title(f'KDE. Q Values. R:{rwd}, Gamma:{gamma}')

def create_all_heatmaps(df_selected_1, df_selected_2, nrows, ncols, rwd_list, gamma_list, smooth):
    _, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5)) # fig size (15, 10) for 2*2

    create_rbf_heatmap(df_selected_1, axs[0], rwd_list[0], gamma_list[0], smooth)
    create_kde_heatmap(df_selected_1, axs[1], rwd_list[0], gamma_list[0])

    create_rbf_heatmap(df_selected_2, axs[2], rwd_list[1], gamma_list[1], smooth)
    create_kde_heatmap(df_selected_2, axs[3], rwd_list[1], gamma_list[1])

    # create_rbf_heatmap(df_selected_1, axs[0, 0], rwd_list[0], gamma_list[0], smooth)
    # create_kde_heatmap(df_selected_1, axs[0, 1], rwd_list[0], gamma_list[0])

    # create_rbf_heatmap(df_selected_2, axs[1, 0], rwd_list[1], gamma_list[1], smooth)
    # create_kde_heatmap(df_selected_2, axs[1, 1], rwd_list[1], gamma_list[1])

    plt.tight_layout()
    plt.show()

def visualize_heatmaps(df_path, unit_rwd_list, rwd_list, gamma_list, m1_col_list, m2_col_list, smooth):
    df = pd.read_csv(df_path)
    df_selected_1 = calculate_Q_values(df, unit_rwd_list[0], rwd_list[0], gamma_list[0], m1_col_list) # module 1
    df_selected_2 = calculate_Q_values(df, unit_rwd_list[1], rwd_list[1], gamma_list[1], m2_col_list) # module 2

    create_all_heatmaps(df_selected_1, df_selected_2, 1, 4, rwd_list, gamma_list, smooth)

def heatmap_vis(results, index, delay):
    rwd1, gamma1, rwd2, gamma2 = results[index].x
    rwd_list = [round(rwd1, 2), round(rwd2, 2)]
    gamma_list = [round(gamma1, 2), round(gamma2, 2)]

    if delay == 0:
        visualize_heatmaps(f't0heatmap/td0_{index+1}_for_heatmap_.csv',
                    unit_rwd_list = [-1,-1], 
                    rwd_list = rwd_list, 
                    gamma_list = gamma_list, 
                    m1_col_list= ['dist11','dist12','dist13','dist14'],
                    m2_col_list= ['dist31', 'dist32'],
                    smooth = 0.1 # RBF smooth value
                    )
    elif delay == 3:
        visualize_heatmaps(f't3heatmap/td3_{index+1}_matched_for_heatmap_.csv',
                    unit_rwd_list = [-1,-1], 
                    rwd_list = rwd_list, 
                    gamma_list = gamma_list, 
                    m1_col_list= ['dist11','dist12','dist13','dist14'],
                    m2_col_list= ['dist31', 'dist32'],
                    smooth = 0.1 # RBF smooth value
                    )

# 사용 예시: visualize_heatmaps('path_to_csv_file', unit_rwd, rwd, gamma, module_column_list)

# def create_interpolated_heatmap(df, method, ax):
#     unique_pD = np.linspace(df.iloc[:, 0].min(), df.iloc[:, 0].max(), 100)
#     unique_tD = np.linspace(df.iloc[:, 1].min(), df.iloc[:, 1].max(), 100)

#     points = df.iloc[:, :2].values
#     values = df['Q'].values

#     grid_x, grid_y = np.meshgrid(unique_pD, unique_tD)
#     grid_z = griddata(points, values, (grid_x, grid_y), method=method)

#     aspect_ratio = (df.iloc[:, 1].max() - df.iloc[:, 1].min()) / (df.iloc[:, 0].max() - df.iloc[:, 0].min())

#     im = ax.imshow(grid_z, cmap='viridis', aspect=aspect_ratio, origin='lower',
#                    extent=[df.iloc[:, 1].min(), df.iloc[:, 1].max(), df.iloc[:, 0].min(), df.iloc[:, 0].max()])
#     plt.colorbar(im, ax=ax, label='Q Value')
#     ax.set_xlabel('Physical Distance')
#     ax.set_ylabel('Temporal Distance')
#     ax.set_title(f'Interpolated Heatmap - {method}')
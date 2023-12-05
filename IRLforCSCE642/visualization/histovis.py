import matplotlib.pyplot as plt

def histo_vis(opt_results_list, logScale1=False, logScale2=False, fontsize=15):
    """
    Visualize histograms of rewards and gamma values from optimization results.

    Args:
    opt_results_list (list): List of optimization result objects.
    logScale1 (bool): Whether to use log scale for rewards. Default is False.
    logScale2 (bool): Whether to use log scale for gamma values. Default is False.
    fontsize (int): Font size for the annotations on the bars.

    This function creates a series of subplots for each set of optimization results.
    For each set, it visualizes rewards and gamma values as bar charts.
    """
    # Create a figure with subplots.
    plt.figure(figsize=(15, 5 * len(opt_results_list)))

    for idx, opt_results in enumerate(opt_results_list):
        # Extract variable values from optimization results.
        rwd1, gamma1, rwd2, gamma2 = opt_results.x

        # Construct lists for rewards and gamma values.
        rwd = [rwd1, rwd2]
        gamma = [gamma1, gamma2]
        modules = ["Use Shortest Path", "Avoid Obstacles"]

        # Plot histogram for rewards.
        plt.subplot(len(opt_results_list), 2, 2 * idx + 1)
        bars = plt.bar(modules, rwd, color=['blue', 'green'], alpha=0.7)
        if logScale1:
            plt.yscale('log')  # Use log scale for rewards if specified.
            plt.ylim(10e-4, 1)
        else:
            plt.ylim(0, 1)
        plt.title(f'Rewards for Each Module - Set {idx+1}')
        plt.ylabel('Reward Value')

        # Annotate the bars with reward values.
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, 
                     height, 
                     f'{height:.2f}', 
                     ha='center', 
                     va='center',
                     fontsize=fontsize)

        # Plot histogram for gamma values.
        plt.subplot(len(opt_results_list), 2, 2 * idx + 2)
        bars = plt.bar(modules, gamma, color=['blue', 'green'], alpha=0.7)
        if logScale2:
            plt.yscale('log')  # Use log scale for gamma if specified.
            plt.ylim(10e-20, 1)
        else:
            plt.ylim(0, 1)
        plt.title(f'Gamma Values for Each Module - Set {idx+1}')
        plt.ylabel('Gamma Value')

        # Annotate the bars with gamma values.
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, 
                     height, 
                     f'{height:.2f}', 
                     ha='center', 
                     va='center',
                     fontsize=fontsize)

    # Display the plots.
    plt.show()

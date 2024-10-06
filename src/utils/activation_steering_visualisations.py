import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_channel_metrics(objectives, channels_to_test=range(31)):
    metrics = ['avg_total_reward', 'avg_steps_until_pickup', 'total_count_pickups']
    titles = ['Average Total Reward', 'Average Steps Until Pickup', 'Total Count of Pickups']
    all_entities = ['keygreen']  # Only 'keygreen' is present in the objectives

    for i, metric in enumerate(metrics):
        plt.figure(figsize=(12, 8))
        
        for entity in all_entities:
            values = []
            for channel in channels_to_test:
                channel_key = f'keygreen_channel{channel}'
                if channel_key in objectives:
                    value = objectives[channel_key][metric]
                    values.append(float(value.item()) if isinstance(value, np.ndarray) else float(value))
                else:
                    values.append(0)  # or np.nan if you prefer to show gaps
            
            # Create a more readable label
            label = f'Green Key'
            
            plt.plot(channels_to_test, values, label=label, marker='o')

        plt.title(titles[i])
        plt.xlabel('Channel Number')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.legend()
        plt.grid(True)
        plt.xticks(range(0, 32, 2))  # Show channel numbers at intervals of 2
        plt.tight_layout()
        plt.show()

        # Optional: If you want to save the plots
        # plt.savefig(f'{metric}_plot.png')
        # plt.close()
import matplotlib.pyplot as plt

def plot_total_pickups_by_channel(objectives):
    plt.figure(figsize=(15, 8))
    
    # Extract unique mod values and channels
    mod_values = sorted(set(int(key.split('_')[1].split('-')[1]) for key in objectives.keys()))
    channels = sorted(set(int(key.split('_')[2].split('channel')[1]) for key in objectives.keys()))
    
    max_pickups = 0  # To keep track of the maximum number of pickups

    for mod in mod_values:
        pickups = []
        for channel in channels:
            key = f'keygreen_mod-{mod}_channel{channel}'
            if key in objectives:
                pickup_count = objectives[key]['total_count_pickups']
                pickups.append(pickup_count)
                max_pickups = max(max_pickups, pickup_count)
            else:
                pickups.append(0)  # or use None to show gaps
        
        plt.plot(channels, pickups, marker='o', linestyle='-', label=f'Mod -{mod}')
    
    plt.title('Total Count of Pickups by Channel for Different Mod Values',fontsize=14)
    plt.xlabel('Channel',fontsize=14)
    plt.ylabel('Total Count of Pickups',fontsize=14)
    plt.legend()
    plt.grid(True)
    
    # Show every 5th channel for readability
    plt.xticks(channels[::3], fontsize=14)
    
    # Set y-axis to integer values from 0 to max_pickups
    plt.yticks(range(101)[::10])
    
    plt.tight_layout()
    plt.show()

def plot_pickup_heatmap(objectives, max_pickups=5):
    # Extract unique mod values and channels
    mod_values = sorted(set(int(key.split('_')[1].split('-')[1]) for key in objectives.keys()), reverse=True)
    channels = sorted(set(int(key.split('_')[2].split('channel')[1]) for key in objectives.keys()))

    # Create a 2D array to hold the pickup counts
    pickup_data = np.zeros((len(mod_values), len(channels)))

    # Fill the array with pickup counts
    for i, mod in enumerate(mod_values):
        for j, channel in enumerate(channels):
            key = f'keygreen_mod-{mod}_channel{channel}'
            if key in objectives:
                pickup_data[i, j] = objectives[key]['total_count_pickups']

    # Calculate appropriate figure size
    fig_width = max(20, len(channels) * 0.3)
    fig_height = len(mod_values) * 1.5

    # Create the heatmap
    plt.figure(figsize=(fig_width, fig_height))
    
    # Custom annotation function
    def annot_fmt(val):
        count = int(val)
        percent = (count / max_pickups) * 100
        return f'{count}\n({percent:.1f}%)'

    ax = sns.heatmap(pickup_data, annot=True, fmt='', cmap='YlOrRd', 
                     xticklabels=channels, yticklabels=mod_values,
                     cbar_kws={'label': 'Total Count of Pickups'},
                     annot_kws={'fontsize': 8},
                     vmin=0, vmax=max_pickups)

    # Add custom annotations
    for i in range(len(mod_values)):
        for j in range(len(channels)):
            text = ax.texts[i * len(channels) + j]
            text.set_text(annot_fmt(pickup_data[i, j]))

    plt.title(f'Heatmap of Total Pickups by Channel and Mod Value (Max: {max_pickups})')
    plt.xlabel('Channel')
    plt.ylabel('Mod Value')
    
    # Improve x-axis readability
    num_xticks = 10
    step = max(1, len(channels) // num_xticks)
    plt.xticks(range(0, len(channels), step), channels[::step], rotation=45, ha='right')
    
    # Adjust y-axis labels
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.show()

def plot_pickup_grouped_bar(objectives):
    # Extract unique mod values and channels
    mod_values = sorted(set(int(key.split('_')[1].split('-')[1]) for key in objectives.keys()), reverse=True)
    channels = sorted(set(int(key.split('_')[2].split('channel')[1]) for key in objectives.keys()))

    # Create a 2D array to hold the pickup counts
    pickup_data = np.zeros((len(mod_values), len(channels)))

    # Fill the array with pickup counts
    for i, mod in enumerate(mod_values):
        for j, channel in enumerate(channels):
            key = f'keygreen_mod-{mod}_channel{channel}'
            if key in objectives:
                pickup_data[i, j] = objectives[key]['total_count_pickups']

    # Set up the plot
    fig, ax = plt.subplots(figsize=(20, 10))

    # Set the width of each bar and the positions of the bars
    bar_width = 0.15
    r = np.arange(len(channels))

    # Plot bars for each mod value
    for i, mod in enumerate(mod_values):
        ax.bar(r + i*bar_width, pickup_data[i], width=bar_width, label=f'Mod -{mod}')

    # Customize the plot
    ax.set_xlabel('Channel')
    ax.set_ylabel('Pickup Count')
    ax.set_title('Grouped Bar Chart of Pickups by Channel and Mod Value')
    ax.set_xticks(r + bar_width * (len(mod_values) - 1) / 2)
    ax.set_xticklabels(channels)
    ax.legend()

    # Show every 5th x-tick label to avoid overcrowding
    for idx, label in enumerate(ax.xaxis.get_ticklabels()):
        if idx % 5 != 0:
            label.set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_pickup_3d_surface(objectives):
    # Extract unique mod values and channels
    mod_values = sorted(set(int(key.split('_')[1].split('-')[1]) for key in objectives.keys()), reverse=True)
    channels = sorted(set(int(key.split('_')[2].split('channel')[1]) for key in objectives.keys()))

    # Create a 2D array to hold the pickup counts
    pickup_data = np.zeros((len(mod_values), len(channels)))

    # Fill the array with pickup counts
    for i, mod in enumerate(mod_values):
        for j, channel in enumerate(channels):
            key = f'keygreen_mod-{mod}_channel{channel}'
            if key in objectives:
                pickup_data[i, j] = objectives[key]['total_count_pickups']

    # Create the 3D surface plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(channels, mod_values)
    surf = ax.plot_surface(X, Y, pickup_data, cmap='viridis', edgecolor='none')

    ax.set_xlabel('Channel')
    ax.set_ylabel('Mod Value')
    ax.set_zlabel('Pickup Count')
    ax.set_title('3D Surface Plot of Pickups by Channel and Mod Value')

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.show()

def plot_pickup_line_graphs(objectives, max_pickups=5):
    # Extract unique mod values and channels
    mod_values = sorted(set(int(key.split('_')[1].split('-')[1]) for key in objectives.keys()), reverse=True)
    channels = sorted(set(int(key.split('_')[2].split('channel')[1]) for key in objectives.keys()))

    for mod in mod_values:
        steering_percentages = []
        for channel in channels:
            key = f'keygreen_mod-{mod}_channel{channel}'
            if key in objectives:
                pickup_count = objectives[key]['total_count_pickups']
                # Calculate steering percentage directly from pickup count
                steering_percentages.append(100 * pickup_count / max_pickups)
            else:
                steering_percentages.append(0)  # 0% if no data

        # Create a new figure for each mod value
        fig, ax = plt.subplots(figsize=(15, 6))

        # Plot the steering percentages
        ax.plot(channels, steering_percentages, marker='o', color='blue')
        ax.set_ylim(100, 0)  # Inverted y-axis
        ax.set_xlabel('Channel')
        ax.set_ylabel('% time steering worked')
        ax.grid(True, linestyle='--', alpha=0.7)

        # Set x-ticks to show every second channel
        ax.set_xticks(channels[::2])
        ax.set_xticklabels(channels[::2])

        # Set y-ticks for percentages
        ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ax.set_yticklabels(['100%', '90%', '80%', '70%', '60%', '50%', '40%', '30%', '20%', '10%', '0%'])  # Inverted labels
        
        # Add minor ticks for more granularity
        ax.set_yticks(range(0, 101, 5), minor=True)
        
        # Customize grid
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray')
        
        # Add horizontal lines at 25%, 50%, and 75%
        for y in [25, 50, 75]:
            ax.axhline(y=y, color='red', linestyle='--', alpha=0.5)

        # Add mod value as a title above the graph
        plt.title(f'Mod -{mod}', pad=20)

        plt.tight_layout()
        plt.savefig(f'mod_{mod}_steering_percentages.png')
        plt.close()

from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib.pyplot as plt

def plot_pickup_line_graphs_smooth(objectives, max_pickups=5, y_min=0, y_max=100):
    # Extract unique mod values and channels
    mod_values = sorted(set(int(key.split('_')[1].split('-')[1]) for key in objectives.keys()), reverse=True)
    channels = sorted(set(int(key.split('_')[2].split('channel')[1]) for key in objectives.keys()))

    # Set up the plot
    plt.figure(figsize=(15, 8))
    
    # Plot data for each mod value
    for mod in mod_values:
        steering_percentages = []
        for channel in channels:
            key = f'keygreen_mod-{mod}_channel{channel}'
            if key in objectives:
                pickup_count = objectives[key]['total_count_pickups']
                steering_percentages.append(100 * pickup_count / max_pickups)
            else:
                steering_percentages.append(0)

        # Smoothing using spline interpolation
        channels_np = np.array(channels)
        steering_np = np.array(steering_percentages)
        if len(channels_np) > 3:  # Need at least 4 points for spline
            spl = make_interp_spline(channels_np, steering_np, k=3)  # Cubic spline
            channels_smooth = np.linspace(channels_np.min(), channels_np.max(), 300)
            steering_smooth = spl(channels_smooth)
            plt.plot(channels_smooth, steering_smooth, label=f'Mod -{mod}')
        else:
            plt.plot(channels, steering_percentages, marker='o', label=f'Mod -{mod}')

    # Customize the plot
    plt.title('Steering Effectiveness by Channel for Different Mod Values', fontsize=16)
    plt.xlabel('Channel', fontsize=12)
    plt.ylabel('% time steering worked', fontsize=12)
    plt.ylim(y_min, y_max)  # Adjusted y-axis
    plt.xticks(range(0, max(channels)+1, 5))  # Show every 5th channel
    plt.yticks(range(y_min, y_max+1, 20))  # Adjusted y-axis ticks
    
    plt.legend(title='Mod Value', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()